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

def gated_attention(doc, qry, inter, mask, gating_fn='tf.multiply', name="Gated_Attention_Layer"):
    # document: B x N x D
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
        # Calculating the actual (document)token-specific representation of the query
        q_rep = tf.matmul(alphas_r, qry)  # B x N x D

        return eval(gating_fn)(doc, q_rep)


def pairwise_interaction(doc, qry, name="Pairwise_Interaction_Layer"):
    # document: B x N x D
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
    # start_pred - B x N
    # end_pred   - B x N
    # Probabilites over document words being the answer start- or end-index

    with tf.name_scope(name):
        # TODO: Make these dense layers dynamic in shape? Add dropout? Different activation?
        start_logits = tf.layers.dense(inputs=inter, units=n_hidden_dense, activation=tf.nn.relu)
        end_logits = tf.layers.dense(inputs=inter, units=n_hidden_dense, activation=tf.nn.relu)

        # Debug, Calculate mean, but ignore zero columns
        # intermediate_tensor1 = tf.reduce_sum(tf.abs(start_logits), axis=1)
        # zero_vector = tf.zeros(shape=(1, 1))
        #
        # mask_columns = tf.squeeze(tf.not_equal(intermediate_tensor1, zero_vector))
        # mask_columns.set_shape([None, ])
        # # Mask both start and end. Since they have the same zero columns.
        # start_logits = tf.boolean_mask(start_logits, mask_columns, axis=2)
        # end_logits = tf.boolean_mask(end_logits, mask_columns, axis=2)
        # End of debug

        start_softmax = tf.nn.softmax(start_logits, axis=1)
        end_softmax = tf.nn.softmax(end_logits, axis=1)

        start_pred = tf.reduce_mean(start_softmax, axis=2)
        end_pred = tf.reduce_mean(end_softmax, axis=2)

        return start_pred, end_pred


def attention_sum_cloze(doc, qry, cand, cloze, cand_mask=None, name="Attention_Sum_Layer"):
    # For cloze-style QA
    # document: B x N x D
    # qry: B x Q x D
    # cand: B x N x C
    # cloze: B x 1
    # cand_mask: B x N
    with tf.name_scope(name):
        # Get the index of the cloze in the query for each sample in batch
        idx = tf.concat(
            [tf.expand_dims(tf.range(tf.shape(qry)[0]), axis=1),
             tf.expand_dims(cloze, axis=1)], axis=1)
        # Retrieve hidden representation of cloze in query for each sample in batch
        q = tf.gather_nd(qry, idx)  # B x D
        # Matrix multiply cloze representation with each document word representation
        p = tf.squeeze(
            tf.matmul(doc, tf.expand_dims(q, axis=-1)), axis=-1)  # B x N
        # Take the softmax of the above and mask it, so it's zero where
        # there is no candidate answer in the document.
        pm = tf.nn.softmax(p) * tf.cast(cand_mask, tf.float32)  # B x N
        # Normalize with the sum of the non-zero entries
        pm = pm / tf.expand_dims(tf.reduce_sum(pm, axis=1), axis=-1)  # B x N
        pm = tf.expand_dims(pm, axis=1)  # B x 1 x N

    # Matrix multiply by the candidate markers
    # This masks out all the non-relevant words in the document, keeping only
    # the candidates. Also keeps them in their original order 1 to C (max_num_cand)
    # Technically allows a candidate to show up multiple times in the document, and
    # adds up the respective softmax values for that candidate.
        return tf.squeeze(
            tf.matmul(pm, tf.cast(cand, tf.float32)), axis=1)  # B x C


def crossentropy(pred, target, name="Cross_Entropy"):
    """
    Cloze-style:
    pred: B x C   - Vector of probabilities of a candidate being the right
    answer among all candidates
    target: B x 1 - The index of the answer among the candidates

    Span-style:
    pred: B x N - Vectors of probabilities of a document word being the answer start or end.
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

