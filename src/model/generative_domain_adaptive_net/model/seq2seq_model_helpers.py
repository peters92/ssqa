#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf


# Bidirectional encoder. Concatenates the output/states from a forward and a backward
# GRU cell. The encoder states are collected from each layer and are used as input to the
# decoder.
def bidirectional_encoder_layer(rnn_cell, n_layers, document_mask, document_embedding,
                                n_hidden_encoder, max_doc_len, keep_probability):

    sequence_length = tf.reduce_sum(document_mask, axis=1)

    encoder_states_list = []

    for i in range(n_layers):
        with tf.variable_scope('encoder_layer_{}'.format(i)):
            encoder_cell_forward = rnn_cell(n_hidden_encoder)
            encoder_cell_backward = rnn_cell(n_hidden_encoder)

            encoder_cell_forward = tf.contrib.rnn.DropoutWrapper(encoder_cell_forward,
                                                                 input_keep_prob=keep_probability)
            encoder_cell_backward = tf.contrib.rnn.DropoutWrapper(encoder_cell_backward,
                                                                  input_keep_prob=keep_probability)

            encoder_output, encoder_states = \
                tf.nn.bidirectional_dynamic_rnn(encoder_cell_forward,
                                                encoder_cell_backward,
                                                document_embedding,
                                                sequence_length,
                                                dtype=tf.float32)

            encoder_states = tf.concat([encoder_states[0], encoder_states[1]], axis=1)
            encoder_states_list.append(encoder_states)

            encoder_output = tf.concat(encoder_output, 2)
            document_embedding = encoder_output

    encoder_states = tuple(encoder_states_list)

    return encoder_output, encoder_states


# Simple encoder, only forward reading of document with a GRU cell
def encoder_layer(rnn_cell, n_layers, document_mask, document_embedding,
                  n_hidden_encoder, max_doc_len, keep_probability):

    encoder_cells = []
    for i in range(n_layers):
        encoder_cell = rnn_cell(n_hidden_encoder)
        encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, input_keep_prob=keep_probability)
        encoder_cells.append(encoder_cell)

    encoder_cells = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)

    sequence_length = tf.reduce_sum(document_mask, axis=1)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cells, document_embedding,
                                                      sequence_length, dtype=tf.float32)

    return encoder_output, encoder_state


def decoder_layer_training(decoder_input, decoder_cells, query_embedding, query_mask,
                           max_query_length, output_layer, keep_probability):
    """
    Decoder layer for training mode. Decodes hidden states received from an encoder layer.
    :return: logits of shape [batch_size, max_query_length, vocabulary_size]
    """
    # TODO: move dropout to decoder_layer and apply for each layer
    # Wrap the input RNN cells with dropout.
    # cells = tf.contrib.rnn.DropoutWrapper(decoder_cells, output_keep_prob=keep_probability)

    # Training Helper. Embeds the target question sequence
    sequence_length = tf.reduce_sum(query_mask, axis=1)
    training_helper = tf.contrib.seq2seq.TrainingHelper(query_embedding, sequence_length)
    # The basic sampling decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cells, training_helper, decoder_input, output_layer)
    # Dynamic decoder, decoding step by step.
    logits, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                     maximum_iterations=max_query_length)

    # TODO: assert shape of logits
    return logits


def decoder_layer_inference(decoder_input, decoder_cells, word_vectors,
                            symbol_begin, symbol_end, max_query_length,
                            output_layer, keep_probability, current_batch_size):
    """
    Decoder layer for inference mode. Decodes hidden states received from an encoder layer, without
    using a target sequence (that is, the query)
    :return: logits of shape [batch_size, max_query_length, vocabulary_size]
    """
    # TODO: move dropout to decoder_layer and apply for each layer
    # Wrap the input RNN cells with dropout.
    # cells = tf.contrib.rnn.DropoutWrapper(decoder_cells, output_keep_prob=keep_probability)

    # Inference helper. Embeds the output of the decoder at each time step
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_vectors,
                                                                tf.fill([current_batch_size], symbol_begin),
                                                                symbol_end)

    # The basic sampling decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cells, inference_helper,
                                              decoder_input, output_layer)
    # Dynamic decoder, decoding step by step.
    logits, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                     maximum_iterations=max_query_length)

    # TODO: assert shape of logits
    return logits


def decoder_layer(encoder_states, encoder_output, query_embedding, query_mask, document_mask,
                  word_vectors, rnn_cell, max_query_length, vocab_size, n_layers, n_hidden_decoder,
                  keep_probability, use_attention, symbol_begin_embedded, symbol_end_embedded,
                  current_batch_size):
    """
    The full decoder layer. Decodes the input (encoder hidden states of embedded document) and returns
    logits over the vocabulary that represent a query given the input paragraph/answer.
    :return: logits of shape [batch_size, max_query_length, vocabulary_size]
    """

    # Defining the N layers for the decoder
    cells = [tf.contrib.rnn.DropoutWrapper(rnn_cell(n_hidden_decoder),
                                           input_keep_prob=keep_probability) for _ in range(n_layers)]
    decoder_cells = tf.nn.rnn_cell.MultiRNNCell(cells)

    # Defining output layer (dense) for calculating logits over the vocabulary
    output_layer = tf.layers.Dense(vocab_size)

    # If true, then use the attention mechanism over the encoder outputs.
    if use_attention:
        sequence_length = tf.reduce_sum(document_mask, axis=1)
        attention = tf.contrib.seq2seq.BahdanauAttention(n_hidden_decoder,
                                                         encoder_output,
                                                         sequence_length,
                                                         name="BahdanauAttentionMechanism")
        decoder_cells = tf.contrib.seq2seq.AttentionWrapper(decoder_cells,
                                                            attention,
                                                            n_hidden_decoder)
        # Initialize zero state
        zero_state = decoder_cells.zero_state(current_batch_size, tf.float32)
        # Copy contents of encoder states into the zero state
        encoder_states = zero_state.clone(cell_state=encoder_states)

    with tf.variable_scope("decode"):
        logits_training = decoder_layer_training(encoder_states, decoder_cells,
                                                 query_embedding, query_mask,
                                                 max_query_length, output_layer,
                                                 keep_probability)

    with tf.variable_scope("decode", reuse=True):
        logits_inference = decoder_layer_inference(encoder_states, decoder_cells,
                                                   word_vectors, symbol_begin_embedded,
                                                   symbol_end_embedded, max_query_length,
                                                   output_layer, keep_probability,
                                                   current_batch_size)

    return logits_training, logits_inference
