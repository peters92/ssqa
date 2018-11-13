#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Dimensions:
# B - Batch size
# N - max. document length in batch
# Q - max. query length in batch
# D - dimensionality of embeddings


def gated_attention(doc, qry, inter,
                    mask, gating_fn='tf.multiply',
                    name="Gated_Attention_Layer"):
    # doc: B x N x D
    # qry: B x Q x D
    # inter: B x N x Q
    # mask (qry): B x Q
    with tf.name_scope(name):
        # Applying softmax over the interaction matrix and then applying
        # the query mask to keep only the relevant values.
        alphas_r = tf.nn.softmax(inter) * \
            tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
        # Normalize values after masking
        alphas_r = alphas_r / \
            tf.expand_dims(tf.reduce_sum(alphas_r, axis=2), axis=-1)  # B x N x Q
        # Calculating the actual (doc)token-specific representation of the query
        q_rep = tf.matmul(alphas_r, qry)  # B x N x D

        return eval(gating_fn)(doc, q_rep)


def pairwise_interaction(doc, qry, name="Pairwise_Interaction_Layer"):
    # doc: B x N x D
    # qry: B x Q x D
    with tf.name_scope(name):
        shuffled = tf.transpose(qry, perm=[0, 2, 1])  # B x D x Q

        return tf.matmul(doc, shuffled)  # B x N x Q


def attention_sum(inter, n_hidden_dense, name="Attention_Sum_Layer"):
    # For span-style QA
    # The pairwise interaction matrix is passed through
    # two fully connected layers with softmax to predict the start and end indices

    # Input:
    # inter - B x N x Q - pairwise interaction matrix
    # Outputs:
    # start_softmax - B x N
    # end_softmax   - B x N
    # Probabilites over document words being the answer start- or end-index

    with tf.name_scope(name):
        # TODO: Make these dense layers dynamic in shape? Add dropout? Different activation?
        # The last axes of inter are multiplied to get N * Q
        desired_shape = inter.get_shape()[-2] * inter.get_shape()[-1]
        # The interaction matrix is reshaped to [B x (N * Q)]
        inter = tf.reshape(inter, [tf.shape(inter)[0], desired_shape])

        # Input to dense layers is [B x (N*Q)]
        # output (with n_hidden_dense = N) shape is [B x N]
        start_softmax = tf.layers.dense(inputs=inter,
                                        units=n_hidden_dense,
                                        activation=tf.nn.softmax)
        end_softmax = tf.layers.dense(inputs=inter,
                                      units=n_hidden_dense,
                                      activation=tf.nn.softmax)

        return start_softmax, end_softmax


def crossentropy(pred, target, name="Cross_Entropy"):
    """
    Span-style:
    pred: B x N   - Vectors of probabilities of a document word being the answer start or end.
    target: B x 1 - The true index of either the answer start or end depending on input
    """
    with tf.name_scope(name):
        # Create vector of batch index concatenated with true index of answer
        idx = tf.expand_dims(tf.range(tf.shape(target)[0]), 1)
        idx = tf.concat([idx, tf.expand_dims(target, 1)], axis=-1)

        # Gather the probability values at the correct answer's index in the prediction
        prob = tf.gather_nd(pred, idx)  # B x 1

        # Return the negative logarithm of the probability
        return - tf.log(prob)

