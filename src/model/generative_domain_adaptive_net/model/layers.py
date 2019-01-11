#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
# =======================================================================
# All references to paper unless below -unless otherwise stated- refer to:
# "Gated-Attention Readers for Text Comprehension" by Dhingra et al. 2016
#   arXiv:  1606.01549v3
#   github: https://github.com/bdhingra/ga-reader
# =======================================================================


def gated_attention(document_embedding, query_embedding, interaction,
                    query_mask, document_mask, gating_fn=tf.multiply):
    # document shape:           [batch_size, max_document_length, embedding_dimensions]
    # query shape:              [batch_size, max_query_length, embedding_dimensions]
    # interaction matrix shape: [batch_size, max_document_length, max_query_length]
    # query_mask:               [batch_size, max_query_length]

    # Applying softmax over the interaction matrix and then applying
    # the query mask to keep only values where there are query words in the example.
    # This is because not all queries are the same length in the batch.
    # shape still: [batch_size, max_document_length, max_query_length]
    # See eq. 5 in paper ()
    alphas_r = tf.nn.softmax(interaction) * \
        tf.cast(tf.expand_dims(query_mask, axis=1), tf.float32)

    # TODO: Try using the document mask as well. Should be completely analogous to query
    # But it has to be expanded on the last axis to fit the shape of alphas_r
    # TEMP
    # alphas_r = alphas_r * \
    #     tf.cast(tf.expand_dims(document_mask, axis=2), tf.float32)
    # TEMP

    # Normalize values after masking by dividing with the sum of the matrix
    # shape still: [batch_size, max_document_length, max_query_length]
    alphas_r = alphas_r / \
        tf.expand_dims(tf.reduce_sum(alphas_r, axis=2), axis=-1)

    # TODO: check that this works
    # Getting rid of NaNs after division by zero
    # small_constant = tf.constant(1e-9, dtype=tf.float32)
    # alphas_r = tf.where(tf.is_nan(alphas_r), tf.zeros_like(alphas_r)+small_constant, alphas_r)

    # Calculating the actual document token-specific representation of the query
    # See second line of eq. 5 in paper
    # shape: [batch_size, max_document_length, embedding_dimensions]
    query_representation = tf.matmul(alphas_r, query_embedding)

    # Final output from element-wise multiplication between document embedding and the
    # document token specific query representation.
    # shape: [batch_size, max_document_length, embedding_dimensions]
    return gating_fn(document_embedding, query_representation)


def pairwise_interaction(document, query):
    # document shape: [batch_size, max_document_length, embedding_dimensions]
    # query shape:    [batch_size, max_query_length, embedding_dimensions]

    # The query is transposed to make it ready for matrix multiplication with
    # the document.
    # Output is shape [batch_size, embedding_dimensions, max_query_length]
    query_transposed = tf.transpose(query, perm=[0, 2, 1])
    # Matrix multiply document and query (transposed). See eq. 5 in paper.
    # Output is shape [batch_size, max_document_length, max_query_length]
    interaction = tf.matmul(document, query_transposed)

    return interaction


def attention_sum(interaction, n_hidden_dense, keep_prob):
    # The pairwise interaction matrix is passed through two fully connected
    # layers with softmax to predict the start and end indices of the answer span.

    # Input:
    # interaction matrix shape: [batch_size, max_document_length, max_query_length]
    # Outputs:
    # start_softmax - [batch_size, max_document_length]
    # end_softmax   - [batch_size, max_document_length]
    # Probabilites over document words being the answer start- or end-index

    # TODO: Make these dense layers dynamic in shape? Add dropout? Different activation?
    # The last axes of the interaction matrix are multiplied to get shape:
    # max_document_length * max_query_length
    desired_shape = interaction.get_shape()[-2] * interaction.get_shape()[-1]
    # The interaction matrix is then reshapes to be shape:
    # [batch_size, (max_document_length * max_query_length)]
    interaction = tf.reshape(interaction, [tf.shape(interaction)[0], desired_shape])

    # Dropout
    interaction = tf.nn.dropout(interaction, keep_prob)

    # Input to dense layers is [batch_size, (max_document_length * max_query_length)]
    # Output is [batch_size, n_hidden_dense] where n_hidden_dense = max_document_length
    start_softmax = tf.layers.dense(inputs=interaction,
                                    units=n_hidden_dense,
                                    activation=tf.nn.softmax)
    end_softmax = tf.layers.dense(inputs=interaction,
                                  units=n_hidden_dense,
                                  activation=tf.nn.softmax)

    return start_softmax, end_softmax


def crossentropy(prediction, target):
    """
    Vectors of probabilities of a document word being the answer start or end.
    prediction: [batch_size, max_document_length]

    The true index of either the answer start or end depending on input
    target: [batch_size, 1]
    """
    # Create vector of batch index concatenated with true index of answer
    idx = tf.expand_dims(tf.range(tf.shape(target)[0]), 1)
    idx = tf.concat([idx, tf.expand_dims(target, 1)], axis=-1)

    # Gather the prediction's probability value at the correct answer's index
    # (for each example in the batch)
    prob = tf.gather_nd(prediction, idx)  # B x 1

    # Return the negative logarithm of the probability
    return - tf.log(prob)


def encoder_layer(rnn_cell, n_layers, document_mask, document_embedding,
                  n_hidden_encoder, max_doc_len, keep_probability):
    for i in range(n_layers):
        # --------------------
        #  BI-DIRECTIONAL GRU
        # --------------------
        # DOCUMENT
        # Define the GRU cells used for the forward and backward document sequence
        forward_document = rnn_cell(n_hidden_encoder)
        backward_document = rnn_cell(n_hidden_encoder)
        # Get the actual length of documents in the current batch
        sequence_length = tf.reduce_sum(document_mask, axis=1)

        # Pass the document through the Bi-GRU (see figure 1 in paper, x_1 to x_T on horizontal arrows)
        (forward_document_states, backward_document_states), _ = \
            tf.nn.bidirectional_dynamic_rnn(
                forward_document, backward_document, document_embedding, sequence_length=sequence_length,
                dtype=tf.float32, scope="layer_{}_doc_rnn".format(i))
        # Concatenate the output from the Bi-GRU, see eq. 1 and 2 in paper
        document_bi_embedding = tf.concat([forward_document_states, backward_document_states], axis=2)

        # Assert shape of document_bi_embedding
        assert document_bi_embedding.shape.as_list() == [None, max_doc_len, 2 * n_hidden_encoder], \
            "Expected document_bi_embedding shape [None, {}, {}] but got {}".format(
                max_doc_len, 2 * n_hidden_encoder, document_bi_embedding.shape)

        # Apply dropout to the document embedding
        document_embedding = tf.nn.dropout(document_bi_embedding, keep_probability)

    return document_embedding
