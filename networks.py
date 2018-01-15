# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf

def TextEnc(L, training=True):
    '''
    Args:
      L: Text inputs. (B, N)

    Return:
        K: Keys. (B, N, d)
        V: Kalues. (B, N, d)
    '''
    i = 1
    tensor = embed(L,
                   vocab_size=hp.vocab_size,
                   num_units=hp.e,
                   scope="embed_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    filters=2*hp.d,
                    size=1,
                    rate=1,
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1

    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            norm_type="ln",
                            dropout_rate=hp.dropout_rate,
                            activation_fn=None,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        norm_type="ln",
                        dropout_rate=hp.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=1,
                        rate=1,
                        norm_type="ln",
                        dropout_rate=hp.dropout_rate,
                        activation_fn=None,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    K, V = tf.split(tensor, 2, -1)
    return K, V

def AudioEnc(S, training=True):
    '''
    Args:
      S: melspectrogram. (B, T/r, n_mels)

    Returns
      Q: Queries. (B, T/r, d)
    '''
    i = 1
    tensor = conv1d(S,
                    filters=hp.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    activation_fn=tf.nn.relu,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    tensor = conv1d(tensor,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for _ in range(2):
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            padding="CAUSAL",
                            norm_type="ln",
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=3,
                        padding="CAUSAL",
                        norm_type="ln",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    # queries = tensor


    return tensor

def Attention(Q, K, V, mononotic_attention=False, prev_max_attentions=None):
    '''
    Args:
      Q: Queries. (B, T/r, d)
      K: Keys. (B, N, d)
      V: Values. (B, N, d)

    Returns:
      R: [Context Vectors; Q]. (B, T/r, 2d)
      A: [Attention]. (B, T/r, N)
    '''
    A = tf.matmul(Q, K, transpose_b=True) * tf.rsqrt(tf.to_float(hp.d))
    if mononotic_attention:  # for inference
        key_masks = tf.sequence_mask(prev_max_attentions, hp.max_N)
        reverse_masks = tf.sequence_mask(hp.max_N - hp.attention_win_size - prev_max_attentions, hp.max_N)[:, ::-1]
        masks = tf.logical_or(key_masks, reverse_masks)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, hp.max_T//hp.r, 1])
        paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
        A = tf.where(tf.equal(masks, False), A, paddings)
    A = tf.nn.softmax(A) # (B, T/r, N)
    max_attentions = tf.argmax(A, -1)  # (B, T/r)
    R = tf.matmul(A, V)
    R = tf.concat((R, Q), -1)

    alignments = tf.transpose(A, [0, 2, 1]) # (B, N, T/r)

    # done_logits (B, T/r, 2)

    # done_logits = tf.layers.dense(R, 2)

    return R, alignments, max_attentions#, done_logits

def AudioDec(R, training=True):
    '''
    Args:
      R: [Context Vectors; Q]. (B, T/r, 2d)

    Returns:
      Y: Melspectrogram predictions. (B, T/r, n_mels)
    '''

    i = 1
    tensor = conv1d(R,
                    filters=hp.d,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for j in range(4):
        tensor = hc(tensor,
                        size=3,
                        rate=3**j,
                        padding="CAUSAL",
                        norm_type="ln",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1

    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        padding="CAUSAL",
                        norm_type="ln",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1
    for _ in range(3):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        padding="CAUSAL",
                        norm_type="ln",
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1
    # mel_hats
    logits = conv1d(tensor,
                    filters=hp.n_mels,
                    size=1,
                    rate=1,
                    padding="CAUSAL",
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    Y = tf.nn.sigmoid(logits) # mel_hats

    # done_logits (B, T/r, 2)
    # done_logits = tf.layers.dense(tensor, 2)
    done_logits = conv1d(R,
                         filters=2,
                         size=1,
                         rate=1,
                         norm_type=None,
                         dropout_rate=0,
                         scope="dones")

    return logits, Y, done_logits

def SSRN(Y, training=True):
    '''
    Args:
      Y: Melspectrogram Predictions. (B, T/r, n_mels)

    Returns:
      Z: Spectrogram Predictions. (B, T, 1+n_fft/2)
    '''
    i = 1 # number of layers

    # -> (B, T/r, c)
    tensor = conv1d(Y,
                    filters=hp.c,
                    size=1,
                    rate=1,
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for j in range(2):
        tensor = hc(tensor,
                      size=3,
                      rate=3**j,
                      norm_type="ln",
                      dropout_rate=hp.dropout_rate,
                      training=training,
                      scope="HC_{}".format(i)); i += 1
    for _ in range(2):
        # -> (B, T/2, c) -> (B, T, c)
        tensor = conv1d_transpose(tensor,
                                  scope="D_{}".format(i),
                                  norm_type="ln",
                                  dropout_rate=hp.dropout_rate,
                                  training=training,); i += 1
        for j in range(2):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            norm_type="ln",
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    # -> (B, T, 2*c)
    tensor = conv1d(tensor,
                    filters=2*hp.c,
                    size=1,
                    rate=1,
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        norm_type="ln",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1
    # -> (B, T, 1+n_fft/2)
    tensor = conv1d(tensor,
                    filters=1+hp.n_fft//2,
                    size=1,
                    rate=1,
                    norm_type="ln",
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1

    for _ in range(2):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        norm_type="ln",
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1
    logits = conv1d(tensor,
               size=1,
               rate=1,
               norm_type="ln",
               dropout_rate=hp.dropout_rate,
               training=training,
               scope="C_{}".format(i))
    Z = tf.nn.sigmoid(logits)
    return logits, Z
