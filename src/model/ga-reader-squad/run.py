#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import os
import logging
import time
from datetime import datetime, date
import numpy as np
from tqdm import tqdm, trange
from utils.DataPreprocessor import DataPreprocessor
from utils.MiniBatchLoader import MiniBatchLoader
from utils.Helpers import check_dir, load_word2vec_embeddings
from data.squad_processing_v11 import squad_parser
from model.GAReader import GAReader

from tensorflow.python import debug as tf_debug


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser(
        description='Gated Attention Reader for \
        Text Comprehension Using TensorFlow')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--resume', type='bool', default=False,
                        help='whether to keep training from previous model')
    parser.add_argument('--use_feat', type='bool', default=False,
                        help='whether to use extra features')
    parser.add_argument('--train_emb', type='bool', default=True,
                        help='whether to train embedding')
    parser.add_argument('--init_test', type='bool', default=False,
                        help='whether to perform initial test')
    parser.add_argument('--model_name', type=str, default="model_{}".format(datetime.now().isoformat()),
                        help='Name of the model, used in saving logs and checkpoints')
    parser.add_argument('--data_dir', type=str, default='/scratch/s161027/ga_reader_data/squad',
                        help='data directory containing input')
    # SQUAD_MOD (Temporary Marker for modification debugging)
    parser.add_argument('--cloze_style', type='bool', default=False,
                        help='Whether the dataset is cloze-style QA. Otherwise it is span-style QA (e.g. SQuAD)')
    # SQUAD_MOD
    parser.add_argument('--log_dir', type=str,
                        default='/scratch/s161027/run_data/ga_reader_squad_test1/log',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str,
                        default='/scratch/s161027/run_data/ga_reader_squad_test1/saved_models',
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
    parser.add_argument('--n_epoch', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--eval_every', type=int, default=10000,
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
    # SQUAD_MOD
    cloze_style = args.cloze_style
    if not cloze_style and not os.path.exists(os.path.join(args.data_dir, "test")):  # Parsing SQuAD dataset
        squad_parser()
    # SQUAD_MOD

    # load data
    dp = DataPreprocessor()
    data = dp.preprocess(
        question_dir=args.data_dir,
        no_training_set=False,
        max_example=args.max_example,
        use_chars=use_chars,
        use_cloze_style=cloze_style,
        only_test_run=False)

    # build minibatch loader
    train_batch_loader = MiniBatchLoader(
        data.training, args.batch_size, sample=1.0, use_cloze_style=cloze_style)
    valid_batch_loader = MiniBatchLoader(
        data.validation, args.batch_size, shuffle=False, use_cloze_style=cloze_style)
    test_batch_loader = MiniBatchLoader(
        data.test, args.batch_size, shuffle=False, use_cloze_style=cloze_style)

    # Fixing the max. document and query length to be the max. of the entire dataset
    max_doc_len = max([train_batch_loader.max_doc_len,
                       valid_batch_loader.max_doc_len,
                       test_batch_loader.max_doc_len])
    max_qry_len = max([train_batch_loader.max_qry_len,
                       valid_batch_loader.max_qry_len,
                       test_batch_loader.max_qry_len])

    if not args.resume:
        logging.info("loading word2vec file ...")
        embed_init, embed_dim = \
            load_word2vec_embeddings(data.dictionary[0], args.embed_file)
        logging.info("embedding dim: {}".format(embed_dim))
        logging.info("initialize model ...")
        model = GAReader(args.n_layers, data.vocab_size, data.num_chars,
                         args.n_hidden, args.n_hidden_dense,
                         embed_dim, args.train_emb,
                         args.char_dim, args.use_feat,
                         args.gating_fn, save_attn=True,
                         use_cloze_style=cloze_style)
        model.build_graph(args.grad_clip, embed_init, args.seed,
                          max_doc_len, max_qry_len, use_cloze_style=cloze_style)
        init = tf.global_variables_initializer()
        loc_init = tf.local_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
    else:
            model = GAReader(args.n_layers, data.vocab_size, data.num_chars,
                         args.n_hidden, 100, args.train_emb,  # TODO: fix embed dim?
                         args.char_dim, args.use_feat, args.gating_fn)

    # Tensorboard
    writer = tf.summary.FileWriter(args.log_dir)
    # Setting GPU memory
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    with sess:
        # Tensorboard
        writer.add_graph(sess.graph)

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
            model.restore(sess, args.save_dir)
            saver = tf.train.Saver(tf.global_variables())
        logging.info('-' * 50)
        lr = args.init_learning_rate
        logging.info("Start training ...")

        epoch_range = tqdm(range(args.n_epoch), leave=True, ascii=True, ncols=100)
        max_it = len(train_batch_loader)

        for epoch in epoch_range:
            start = time.time()
            it = loss = acc = n_example = 0
            if epoch >= 2:
                lr /= 2

            for training_data in train_batch_loader:
                loss_, acc_, updates_, attentions_, pred_, answer_ = \
                    model.train(sess, training_data, args.drop_out, lr, it, writer, epoch, max_it)

                # SQUAD_MOD
                attentions_ = np.array(attentions_)
                attentions_nonzero = attentions_[-1, -1, :, :]
                attentions_nonzero = \
                    attentions_nonzero[~np.all(attentions_nonzero==0, axis=1)][:, ~np.all(attentions_nonzero==0, axis=0)]
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
                    statement = "Size of model attentions: {}".format(attentions_.shape)
                    statement += "\nFull attention matrix:\n" + str(attentions_[-1, -1, :, :])
                    statement += "\nNon-zero part of attention matrix (shape: {}):".format(attentions_nonzero.shape) +\
                                 str(attentions_nonzero)
                    # SQUAD_MOD
                    statement += "\nEpoch: {}, it: {} (max: {}), " \
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
                # save model
                if it % args.eval_every == 0 or \
                        it % max_it == 0:
                    valid_loss, valid_acc = model.validate(
                        sess, valid_batch_loader)
                    if valid_acc >= best_acc:
                        logging.info("Best valid acc: {:.3f}, previous best: {:.3f}".format(
                            valid_acc,
                            best_acc))
                        best_acc = valid_acc
                        model.save(sess, saver, args.save_dir, epoch)
                    start = time.time()
        # test model
        logging.info("Final test ...")
        model.validate(sess, test_batch_loader)


if __name__ == "__main__":
    args = get_args()
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    # check the existence of directories
    args.data_dir = os.path.join(os.getcwd(), args.data_dir)
    check_dir(args.data_dir, exit_function=True)
    args.log_dir = os.path.join(os.getcwd(), args.log_dir)
    args.save_dir = os.path.join(os.getcwd(), args.save_dir)
    check_dir(args.log_dir, args.save_dir, exit_function=False)
    # initialize log file
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
