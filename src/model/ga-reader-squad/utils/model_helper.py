#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf


def gated_attention(doc, qry, inter, mask, gating_fn='tf.multiply', name="Gated_Attention_Layer"):
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
    # doc: B x N x D
    # qry: B x Q x D
    # After pairwise interaction the doc/qry interaction is passed through
    # two fully connected layers with softmax to predict the start and end indices
    with tf.name_scope(name):
        # Old, interaction matrix calculated inside the method
        # qry_perm = tf.transpose(qry, perm=[0, 2, 1])  # B x D x Q
        # p = tf.matmul(doc, qry_perm)  # B x N x Q

        start_logits = tf.layers.dense(inputs=inter, units=n_hidden_dense, activation=tf.nn.relu)
        end_logits = tf.layers.dense(inputs=inter, units=n_hidden_dense, activation=tf.nn.relu)

        start_softmax = tf.nn.softmax(start_logits, dim=0)
        end_softmax = tf.nn.softmax(end_logits, dim=0)

        start_pred = tf.reduce_mean(start_softmax, axis=1)
        end_pred = tf.reduce_mean(end_softmax, axis=1)

        return start_pred, end_pred


def attention_sum_cloze(doc, qry, cand, cloze, cand_mask=None, name="Attention_Sum_Layer"):
    # For cloze-style QA
    # doc: B x N x D
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
        # Matrix multiply cloze representation with each doc word representation
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
    # Technically allows a candidate to show up multiple times in the doc, and
    # adds up the respective softmax values for that candidate.
        return tf.squeeze(
            tf.matmul(pm, tf.cast(cand, tf.float32)), axis=1)  # B x C


def crossentropy(pred, target, name="Cross_Entropy"):
    """
    pred: B x C   - Vector of probabilities of a candidate being the right
    answer among all candidates
    target: B x 1 - The index of the answer among the candidates
    """
    with tf.name_scope(name):
        # Masking out zero columns
        intermediate_tensor1 = tf.reduce_sum(tf.abs(pred), axis=0)
        zero_vector = tf.zeros(shape=(1, 1))

        mask_columns = tf.squeeze(tf.not_equal(intermediate_tensor1, zero_vector))
        mask_columns.set_shape([None])
        pred = tf.boolean_mask(pred, mask_columns, axis=1)

        idx = tf.expand_dims(tf.range(tf.shape(target)[0]), 1)
        idx = tf.concat([idx, tf.expand_dims(target, 1)], axis=-1)

        logit = tf.gather_nd(pred, idx)  # B x 1

        return - tf.log(logit)

