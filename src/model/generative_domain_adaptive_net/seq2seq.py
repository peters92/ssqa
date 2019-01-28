#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle
import os
import argparse
import logging
import time
from datetime import datetime
from tqdm import tqdm

from utils.DataPreprocessor import DataPreprocessor
from utils.MiniBatchLoader import MiniBatchLoader
from utils.Helpers import check_dir, load_word2vec_embeddings, batch_splitter
from model.seq2seq_model import Seq2Seq


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(
        description='Gated Attention Reader for \
        Text Comprehension Using TensorFlow')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--resume', type='bool', default=False,
                        help='whether to keep training from previous model')
    parser.add_argument('--test_only', type='bool', default=False,
                        help='whether to only do test run')
    parser.add_argument('--use_feat', type='bool', default=False,
                        help='whether to use question-evidence common feature')
    parser.add_argument('--train_emb', type='bool', default=False,
                        help='whether to train embedding')
    parser.add_argument('--init_test', type='bool', default=False,
                        help='whether to perform initial test')
    parser.add_argument('--model_name', type=str, default="model_{}".format(datetime.now().isoformat()),
                        help='Name of the model, used in saving logs and checkpoints')
    parser.add_argument('--data_dir', type=str, default='/scratch/s161027/ga_reader_data/ssqa_processed',
                        help='data directory containing input')
    parser.add_argument('--training_set', type=str,
                        default='0.9',
                        help='Which training set to use: 0.1, 0.2, 0.5 or 0.9 (complete set)')
    parser.add_argument('--unlabeled_set', type=str,
                        default='small',
                        help='Which unlabeled set to use: small (50k) or large (5m)')
    parser.add_argument('--log_dir', type=str,
                        default='/scratch/s161027/run_data/SSQA/COMBINED_RUNS/log',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str,
                        default='/scratch/s161027/run_data/SSQA/COMBINED_RUNS/saved_models',
                        help='directory to store checkpointed models')
    parser.add_argument('--embed_file', type=str,
                        default='/scratch/s161027/ga_reader_data/word2vec_glove.txt',
                        help='word embedding initialization file')
    parser.add_argument('--n_hidden', type=int, default=256,
                        help='size of word GRU hidden state')
    parser.add_argument('--n_hidden_dense', type=int, default=1024,
                        help='size of final dense layer')
    # =================== SEQ2SEQ =======================================================
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Size of desired vocabulary. This will limit the vocab to the N most\''
                        'frequent words out of all words found in the data. The rest are "@unk".')
    parser.add_argument('--gen_vocab_size', type=int, default=10000,
                        help='Size of Copy Mechanism vocabulary. This will allow only the first '
                        'gen_vocab_size from the vocab to be copied from the source document.')
    parser.add_argument('--n_hidden_encoder', type=int, default=200,
                        help='size of seq2seq encoder layer')
    parser.add_argument('--n_hidden_decoder', type=int, default=200,
                        help='size of seq2seq decoder layer')
    parser.add_argument('--answer_injection', type=bool, default=True,
                        help='Whether or not to inject answer information into document embedding')
    parser.add_argument('--bi_encoder', type=bool, default=True,
                        help='Whether or not to use a bidirectional encoder')
    parser.add_argument('--use_attention', type=bool, default=True,
                        help='Whether or not to use attention over the encoder outputs')
    parser.add_argument('--use_copy_mechanism', type=bool, default=True,
                        help='Whether or not to use the copy mechanism')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of layers of the model')
    parser.add_argument('--max_parallel_dec', type=int, default=8,
                        help='Maximum parallel iterations in while loop of decoder. This helps to'
                             'manage GPU memory.')
    parser.add_argument('--use_cudnn_gru', type=bool, default=False,
                        help='Whether or not to use the CUDNN compatible GRU cell.')
    # =================== SEQ2SEQ =======================================================
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size')
    parser.add_argument('--n_epoch', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--eval_per_epoch', type=int, default=2,
                        help='Evaluation frequency. 1 = once per epoch, at the end of the epoch.')
    parser.add_argument('--print_every', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip gradients at this value')
    parser.add_argument('--init_learning_rate', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for tensorflow')
    parser.add_argument('--max_example', type=int, default=100,
                        help='maximum number of training examples')
    parser.add_argument('--char_dim', type=int, default=512,
                        help='size of character GRU hidden state')
    parser.add_argument('--gating_fn', type=str, default='tf.multiply',
                        help='gating function')
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help='dropout rate')
    args = parser.parse_args()
    return args


def train(args):
    use_chars = args.char_dim > 0
    # Initialize session early to reserve GPU memory
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session()

    # Processing .question files and loading into memory (data in lists and tuples)
    dp = DataPreprocessor()
    data = dp.preprocess(
        question_dir=args.data_dir,
        unlabeled_set=args.unlabeled_set,
        training_set=args.training_set,
        vocab_size=args.vocab_size,
        max_example=args.max_example,
        use_chars=use_chars,
        only_test_run=args.test_only)

    # Define the reverse word dictionary for inference
    word_dict = data.dictionary[0]
    inverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))

    # Building the iterable batch loaders (data in numpy arrays)
    train_batch_loader = MiniBatchLoader(
        data.training, args.batch_size, word_dict, shuffle=True, sample=1.0)
    valid_batch_loader = MiniBatchLoader(
        data.validation, args.batch_size, word_dict, shuffle=False)
    test_batch_loader = MiniBatchLoader(
        data.test, args.batch_size, word_dict, shuffle=False)

    # Fixing the max. document and query length
    # Currently the max. is the same across all batches
    max_doc_len = max([train_batch_loader.max_document_length,
                       valid_batch_loader.max_document_length,
                       test_batch_loader.max_document_length])
    max_qry_len = max([train_batch_loader.max_query_length,
                       valid_batch_loader.max_query_length,
                       test_batch_loader.max_query_length])

    if not args.resume:
        # Loading the GLoVE vectors
        logging.info("loading word2vec file ...")
        embed_init, embed_dim = \
            load_word2vec_embeddings(data.dictionary[0], args.embed_file)
        logging.info("embedding dim: {}".format(embed_dim))
        logging.info("initialize model ...")
        model = Seq2Seq(args.n_layers, data.dictionary, data.vocab_size, args.n_hidden_encoder,
                        args.n_hidden_decoder, embed_dim, args.train_emb, args.answer_injection,
                        args.batch_size, args.bi_encoder, args.use_attention,
                        args.use_copy_mechanism, args.max_parallel_dec, args.gen_vocab_size,
                        args.use_cudnn_gru)
        model.build_graph(args.grad_clip, embed_init, args.seed,
                          max_doc_len, max_qry_len)
        print("\n\nModel build successful!\n\n")
        init = tf.global_variables_initializer()
        loc_init = tf.local_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
    else:
        pass

    with sess:
        sess.graph.finalize()

        # Tensorboard. Two writers to have training and validation accuracy both on the same
        # board.
        writer = tf.summary.FileWriter(args.log_dir+"/tensorboard/training",
                                       graph=sess.graph,
                                       flush_secs=60,
                                       filename_suffix="_train_"+args.model_name)
        writer_val = tf.summary.FileWriter(args.log_dir+"/tensorboard/validation",
                                           graph=sess.graph,
                                           flush_secs=60,
                                           filename_suffix="_validate_"+args.model_name)

        # training phase
        if not args.resume:
            sess.run([init, loc_init])
            if args.init_test:
                logging.info('-' * 50)
                logging.info("Initial test ...")
                best_loss, best_perplexity = model.validate(sess, valid_batch_loader)
            else:
                # Initialize the best validation accuracy to be 0
                best_perplexity = 1e6
        else:
            model.restore(sess, args.save_dir, epoch=9)
            saver = tf.train.Saver(tf.global_variables())

        logging.info('-' * 100)
        logging.info("Start training ...")

        learning_rate = args.init_learning_rate
        epoch_range = tqdm(range(args.n_epoch), leave=True, ascii=True, ncols=100)
        max_it = len(train_batch_loader)  # Max number of batches in training epoch
        eval_every = max_it//args.eval_per_epoch  # How often to validate
        eval_list = list(range(eval_every, max_it+1, eval_every))  # List of validation points
        eval_list[-1] = max_it  # To make sure validation happens exactly at the end of the epoch

        # Define where run statistics will be saved (training loss, validation loss etc.)
        dump_name = args.log_dir + "/run_stats_"+args.model_name

        # Initialize lists for pickle dump of stats
        train_dict_dump = {"train_iteration": [],
                           "train_loss": [],
                           "train_perplexity": [],
                           "epoch": []}
        valid_dict_dump = {"train_iteration": [],
                           "valid_loss": [],
                           "valid_perplexity": [],
                           "epoch": []}

        # Start training loop
        for epoch in epoch_range:
            start_time = time.time()
            # Initialize counters at start of epoch
            it = loss = num_example = 0

            # From the 2. epoch and onwards we adjust the learning rate
            # and re-initialize the accuracy metric
            if epoch >= 2:
                # Halve learning rate
                learning_rate /= 2

            # Count amount of times batches had to be split
            split_count = 0
            # Count amount of times a batch had to be skipped because it exhausted GPU memory
            skip_count = 0

            training_batch_range = tqdm(train_batch_loader, leave=False, ascii=True, ncols=100)
            for training_data in training_batch_range:

                # Check the total sequence length in the current batch. If it's above a certain
                # limit, then split the batch in two
                total_sequence_length = np.sum(training_data[5])

                try:
                    if total_sequence_length > 8700:  # Current GPU (GTX 1080) fails at 8737
                        split_count += 1
                        # print("\nTotal sequence length in batch is too high ({}), "
                        #       "splitting batch in two. Splits performed so far: {}".format(
                        #        total_sequence_length, split_count))

                        batch_split_1, batch_split_2 = batch_splitter(training_data)
                        output_split_1 = \
                            model.train(sess, batch_split_1, args.drop_out, learning_rate, it,
                                        writer,
                                        epoch, max_it)
                        output_split_2 = \
                            model.train(sess, batch_split_2, args.drop_out, learning_rate, it,
                                        writer,
                                        epoch, max_it)

                        loss_ = 0.5 * (output_split_1[0] + output_split_2[0])

                    else:  # Pass the batch to the training method regularly
                        loss_, updates_ = \
                            model.train(sess, training_data, args.drop_out,
                                        learning_rate, it, writer, epoch, max_it)

                except tf.errors.ResourceExhaustedError:
                    skip_count += 1
                    print("GPU out of memory. Total sequence length in batch was {}."
                          "Skipping batch ({} skips so far)..."
                          .format(total_sequence_length, skip_count))
                    continue

                it += 1

                # Cumulative loss, perplexity
                loss += loss_

                # Saving loss, accuracy and iteration for later pickle dump
                train_dict_dump["train_loss"].append(loss_)
                train_dict_dump["train_perplexity"].append(np.exp(loss_))
                train_dict_dump["train_iteration"].append(epoch * max_it + it - 1)
                train_dict_dump["epoch"].append(epoch)

                # Adding the number of examples in the current batch
                num_example += training_data[0].shape[0]

                # Logging information
                if it % args.print_every == 0 or \
                   it == max_it:

                    time_spent = (time.time() - start_time) / 60
                    # Get estimated finish time in hours
                    eta = (time_spent / 60) * ((max_it - it) / args.print_every)

                    # Calculate current perplexity
                    current_loss = loss/args.print_every
                    current_perplexity = np.exp(current_loss)

                    statement = "Epoch: {}, it: {} (max: {}), " \
                        .format(epoch, it, max_it)
                    statement += "Loss: {:.3f}, Perplexity: {:.3f}, " \
                        .format(current_loss,
                                current_perplexity)
                    statement += "Time: {:.1f}(m), " \
                        .format(time_spent)
                    statement += "ETA: {:.1f} hours" \
                        .format(eta)
                    logging.info(statement)
                    loss = num_example = 0
                    start_time = time.time()

                # Validate, and save model
                if it in eval_list:
                    logging.info("{:-^80}".format(" Validation "))

                    valid_loss, valid_perplexity = \
                        model.validate(sess, valid_batch_loader,
                                       inverse_word_dict, it,
                                       writer_val, epoch, max_it)

                    valid_dict_dump["valid_loss"].append(valid_loss)
                    valid_dict_dump["valid_perplexity"].append(valid_perplexity)
                    valid_dict_dump["train_iteration"].append(epoch * max_it + it - 1)
                    valid_dict_dump["epoch"].append(epoch)

                    # Saving run statistics
                    logging.info("Saving run statistics in binary...")
                    print("Saving run statistics in binary...")
                    outfile = open(dump_name, "wb")
                    pickle.dump([train_dict_dump, valid_dict_dump], outfile)
                    outfile.close()

                    if valid_perplexity <= best_perplexity:
                        logging.info("Best validation perplexity: {:.3f}, "
                                     "previous best: {:.3f}".format(
                                      valid_perplexity,
                                      best_perplexity))
                        best_perplexity = valid_perplexity
                        best_epoch = epoch

                        print("The model reached a new best perplexity: {}, at epoch {}"
                              .format(best_perplexity, best_epoch))

                        model.save(sess, saver, args.save_dir, args.model_name, epoch)
                    start_time = time.time()

        # test model
        logging.info("{:-^80}".format(" Final Test "))

        try:
            model.validate(sess, test_batch_loader, inverse_word_dict)
        except tf.errors.ResourceExhaustedError:
            print("GPU out of memory during testing. Skipping batch...")

        # Save run statistics
        logging.info("Saving run statistics in binary...")
        print("Saving run statistics in binary...")
        outfile = open(dump_name, "wb")
        pickle.dump([train_dict_dump, valid_dict_dump], outfile)
        outfile.close()

        # TODO: Send e-mail with run information. Loss, epoch, validation inference example etc.

        input("Script ready to finish, please press enter to exit...")


if __name__ == "__main__":
    args = get_args()
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    # Check the existence of directories
    args.data_dir = os.path.join(os.getcwd(), args.data_dir)
    check_dir(args.data_dir, exit_function=True)
    args.log_dir = os.path.join(os.getcwd(), args.log_dir)
    args.save_dir = os.path.join(os.getcwd(), args.save_dir)
    check_dir(args.log_dir, args.save_dir, exit_function=False)
    # Initialize log file
    current_time = datetime.now().isoformat()

    log_file = os.path.join(args.log_dir, '{}.log'.format(args.model_name))
    if args.log_dir is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    logging.info(args)
    train(args)
