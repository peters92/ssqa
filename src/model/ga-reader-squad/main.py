#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import argparse
import logging
import time
from datetime import datetime
from tqdm import tqdm

from utils.DataPreprocessor import DataPreprocessor
from utils.MiniBatchLoader import MiniBatchLoader
from utils.Helpers import check_dir, load_word2vec_embeddings
from model.GAReader import GAReader
from data.squad_processing_v11 import squad_parser


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
    parser.add_argument('--train_emb', type='bool', default=True,
                        help='whether to train embedding')
    parser.add_argument('--init_test', type='bool', default=False,
                        help='whether to perform initial test')
    parser.add_argument('--model_name', type=str, default="model_{}".format(datetime.now().isoformat()),
                        help='Name of the model, used in saving logs and checkpoints')
    parser.add_argument('--data_dir', type=str, default='/scratch/s161027/ga_reader_data/squad',
                        help='data directory containing input')
    parser.add_argument('--log_dir', type=str,
                        default='/scratch/s161027/run_data/visualization_test/log',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str,
                        default='/scratch/s161027/run_data/visualization_test/saved_models',
                        help='directory to store checkpointed models')
    parser.add_argument('--embed_file', type=str,
                        default='/scratch/s161027/ga_reader_data/word2vec_glove.txt',
                        help='word embedding initialization file')
    parser.add_argument('--n_hidden', type=int, default=256,
                        help='size of word GRU hidden state')
    parser.add_argument('--n_hidden_dense', type=int, default=1024,
                        help='size of final dense layer')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of layers of the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--eval_every', type=int, default=2000,
                        help='evaluation frequency')
    parser.add_argument('--print_every', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip gradients at this value')
    parser.add_argument('--init_learning_rate', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for tensorflow')
    parser.add_argument('--max_example', type=int, default=None,
                        help='maximum number of training examples')
    parser.add_argument('--char_dim', type=int, default=0,
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
    sess = tf.Session()

    if not os.path.exists(os.path.join(args.data_dir, "training")):
        # Parsing SQuAD data set
        squad_parser()

    # Processing .question files and loading into memory (data in lists and tuples)
    dp = DataPreprocessor()
    data = dp.preprocess(
        question_dir=args.data_dir,
        max_example=args.max_example,
        use_chars=use_chars,
        only_test_run=args.test_only)

    # Building the iterable batch loaders (data in numpy arrays)
    train_batch_loader = MiniBatchLoader(
        data.training, args.batch_size, sample=1.0)
    valid_batch_loader = MiniBatchLoader(
        data.validation, args.batch_size, shuffle=False)
    test_batch_loader = MiniBatchLoader(
        data.test, args.batch_size, shuffle=False)

    # Fixing the max. document and query length
    # Currently the max. is the same across all batches
    max_doc_len = max([train_batch_loader.max_doc_len,
                       valid_batch_loader.max_doc_len,
                       test_batch_loader.max_doc_len])
    max_qry_len = max([train_batch_loader.max_qry_len,
                       valid_batch_loader.max_qry_len,
                       test_batch_loader.max_qry_len])

    if not args.resume:
        # Loading the GLoVE vectors
        logging.info("loading word2vec file ...")
        embed_init, embed_dim = \
            load_word2vec_embeddings(data.dictionary[0], args.embed_file)
        logging.info("embedding dim: {}".format(embed_dim))
        logging.info("initialize model ...")
        model = GAReader(args.n_layers, data.vocab_size, data.num_chars,
                         args.n_hidden, args.n_hidden_dense,
                         embed_dim, args.train_emb,
                         args.char_dim, args.use_feat,
                         args.gating_fn, save_attn=True)
        model.build_graph(args.grad_clip, embed_init, args.seed,
                          max_doc_len, max_qry_len)
        init = tf.global_variables_initializer()
        loc_init = tf.local_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
    else:
        model = GAReader(args.n_layers, data.vocab_size, data.num_chars,
                         args.n_hidden, args.n_hidden_dense, 100, args.train_emb, args.char_dim,
                         args.use_feat, args.gating_fn)

    # Setting GPU memory
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)

    with sess:
        # Tensorboard
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
                best_loss, best_acc = model.validate(sess, valid_batch_loader)
            else:
                best_acc = 0.
        else:
            model.restore(sess, args.save_dir, epoch=9)
            saver = tf.train.Saver(tf.global_variables())

        logging.info('-' * 100)
        logging.info("Start training ...")

        lr = args.init_learning_rate
        epoch_range = tqdm(range(args.n_epoch), leave=True, ascii=True, ncols=100)
        max_it = len(train_batch_loader)

        # Start training loop
        for epoch in epoch_range:
            start = time.time()
            it = loss = acc = n_example = 0
            if epoch >= 2:
                lr /= 2

                # Reinitialize streaming accuracy metric for each epoch
                running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                                 scope="accuracy_metric")

                print("\n\nDEBUG MESSAGE:")
                print("TF variables with scope='accuracy_metric': {}".format(
                      running_vars))

                running_vars_initializer = tf.variables_initializer(var_list=running_vars)
                sess.run(running_vars_initializer)

            for training_data in train_batch_loader:
                loss_, acc_, updates_ = \
                    model.train(sess, training_data, args.drop_out, lr, it, writer, epoch, max_it)

                # SQUAD_MOD
                # attentions_ = np.array(attentions_)
                # attentions_nonzero = attentions_[-1, -1, :, :]
                # attentions_nonzero = \
                #     attentions_nonzero[~np.all(attentions_nonzero==0, axis=1)][:, ~np.all(attentions_nonzero==0, axis=0)]
                # SQUAD_MOD

                loss += loss_
                acc += acc_
                it += 1
                n_example += training_data[0].shape[0]

                if it % args.print_every == 0 or \
                        it % max_it == 0:
                    spend = (time.time() - start) / 60
                    # Get estimated finish time in hours
                    eta = (spend / 60) * ((max_it - it) / args.print_every)

                    # SQUAD_MOD
                    # statement = "Size of model attentions: {}".format(attentions_.shape)
                    # statement += "\nFull attention matrix:\n" + str(attentions_[-1, -1, :, :])
                    # statement += "\nNon-zero part of attention matrix (shape: {}):".format(attentions_nonzero.shape) +\
                    #              str(attentions_nonzero)
                    # SQUAD_MOD
                    statement = "Epoch: {}, it: {} (max: {}), " \
                        .format(epoch, it, max_it)
                    statement += "loss: {:.3f}, acc: {:.3f}, " \
                        .format(loss / args.print_every,
                                acc / n_example)
                    statement += "time: {:.1f}(m), " \
                        .format(spend)
                    statement += "ETA: {:.1f} hours" \
                        .format(eta)
                    logging.info(statement)
                    loss = acc = n_example = 0
                    start = time.time()
                # Validate, and save model
                if it % args.eval_every == 0 or \
                        it % max_it == 0:
                    # Reinitialize streaming accuracy metric for each validation
                    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                                     scope="valid_accuracy_metric")
                    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
                    sess.run(running_vars_initializer)

                    valid_loss, valid_acc = \
                        model.validate(sess, valid_batch_loader, it, writer_val, epoch, max_it)
                    if valid_acc >= best_acc:
                        logging.info("Best valid acc: {:.3f}, previous best: {:.3f}".format(
                            valid_acc,
                            best_acc))
                        best_acc = valid_acc
                        # model.save(sess, saver, args.save_dir, epoch)
                    start = time.time()
            # Save model at end of epoch
            model.save(sess, saver, args.save_dir, epoch)
        # test model
        logging.info("Final test ...")
        model.validate(sess, test_batch_loader)


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
