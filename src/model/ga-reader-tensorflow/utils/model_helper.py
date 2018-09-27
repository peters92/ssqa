#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf


def tfconcat(t1, t2):
    return tf.concat([t1, t2], axis=2)


def tfsum(t1, t2):
    return t1 + t2


def gated_attention(doc, qry, inter, mask, gating_fn='tf.multiply', name="Gated_Attention_Layer"):
    # doc: B x N x D
    # qry: B x Q x D
    # inter: B x N x Q
    # mask (qry): B x Q
    with tf.name_scope(name):
        alphas_r = tf.nn.softmax(inter) * \
            tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
        alphas_r = alphas_r / \
            tf.expand_dims(tf.reduce_sum(alphas_r, axis=2), axis=-1)  # B x N x Q
        q_rep = tf.matmul(alphas_r, qry)  # B x N x D

    return eval(gating_fn)(doc, q_rep)


def pairwise_interaction(doc, qry, name="Pairwise_Interaction_Layer"):
    # doc: B x N x D
    # qry: B x Q x D
    with tf.name_scope(name):
        shuffled = tf.transpose(qry, perm=[0, 2, 1])  # B x D x Q

    return tf.matmul(doc, shuffled)  # B x N x Q


def attention_sum(doc, qry, cand, cloze, cand_mask=None, name="Attention_Sum_Layer"):
    # doc: B x N x D
    # qry: B x Q x D
    # cand: B x N x C
    # cloze: B x 1
    # cand_mask: B x N
    with tf.name_scope(name):
        idx = tf.concat(
            [tf.expand_dims(tf.range(tf.shape(qry)[0]), axis=1),
             tf.expand_dims(cloze, axis=1)], axis=1)
        q = tf.gather_nd(qry, idx)  # B x D
        p = tf.squeeze(
            tf.matmul(doc, tf.expand_dims(q, axis=-1)), axis=-1)  # B x N
        pm = tf.nn.softmax(p) * tf.cast(cand_mask, tf.float32)  # B x N
        pm = pm / tf.expand_dims(tf.reduce_sum(pm, axis=1), axis=-1)  # B x N
        pm = tf.expand_dims(pm, axis=1)  # B x 1 x N

    return tf.squeeze(
        tf.matmul(pm, tf.cast(cand, tf.float32)), axis=1)  # B x C


def crossentropy(pred, target, name="Cross_Entropy"):
    """
    pred: B x C
    target: B x 1
    """
    with tf.name_scope(name):
        idx = tf.expand_dims(tf.range(tf.shape(target)[0]), 1)
        idx = tf.concat([idx, tf.expand_dims(target, 1)], axis=-1)
        logit = tf.gather_nd(pred, idx)  # B x 1

    return - tf.log(logit)
