# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

from hparams import Hyperparams as hp
from modules import *
import tensorflow as tf


def encoder(x):
    '''
    :param x: waveform. [B, T, 1]
    :return: z_e: encoded variable. [B, T', D]
    '''
    for i in range(hp.encoder_layers):
        x = conv1d(x,
                    filters=hp.d,
                    size=hp.winsize,
                    strides=hp.stride,
                    padding="valid",
                    bn=True,
                    activation_fn=tf.nn.relu if i < hp.encoder_layers-1 else None)
    z_e = x
    return z_e

def vq(z_e):
    '''

    :param z_e: encoded variable. [B, T', D].
    :return: z_q: nearest embeddings. [B, T', D].
    '''
    lookup_table = tf.get_variable('lookup_table',
                                   dtype=tf.float32,
                                   shape=[hp.K, hp.D],
                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    z = tf.expand_dims(z_e, -2) # (B, T', 1, D)
    lookup_table = tf.reshape(lookup_table, [1, 1, hp.K, hp.D]) # (1, 1, K, D)
    dist = tf.norm(z - lookup_table, axis=-1) # Broadcasting -> (B, T', K)
    k = tf.argmin(dist, axis=-1) # (B, T')
    z_q = tf.gather(lookup_table, k) # (B, T', D)

    return z_q

def decoder()


num_blocks = 3     # dilated blocks
num_dim = 128      # latent dimension

def wavenet()
    def residual_block(inputs, size, rate, scope="res_block", reuse=None):
        with tf.variable_scope(scope=scope, reuse=reuse):
            conv = conv1d(inputs, size=size, rate=rate, activation_fn=tf.tanh, bn=True, scope="conv")
            gate = conv1d(inputs, size=size, rate=rate, activation_fn=tf.sigmoid, bn=True, scope="gate")
            conv *= gate
            outputs = conv1d(conv, size=1, scope="one_by_one")
        return outputs + inputs, outputs

    # dilated conv block loop
    skip = 0  # skip connections
    for i in range(num_blocks):
        for r in (1, 2, 4, 8, 16):
            z, s = residual_block(z, size=7, rate=r, scope="res_block_{}".format(r))
            skip += s

    skip = tf.nn.relu(skip)
    skip = conv1d(skip, activation_fn=tf.nn.relu, scope="one_by_one_1")
    logits = conv1d(skip, filters=len(hp.vocab), scope="one_by_one_2")





def get_logit(x, voca_size):

    # residual block
    def res_block(tensor, size, rate, block, dim=num_dim):

        with tf.sg_context(name='block_%d_%d' % (block, rate)):

            # filter convolution
            conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True, name='conv_filter')

            # gate convolution
            conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True, name='conv_gate')

            # output by gate multiplying
            out = conv_filter * conv_gate

            # final output
            out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, name='conv_out')

            # residual and skip output
            return out + tensor, out

    # expand dimension
    with tf.sg_context(name='front'):
        z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, name='conv_in')

    # dilated conv block loop
    skip = 0  # skip connections
    for i in range(num_blocks):
        for r in [1, 2, 4, 8, 16]:
            z, s = res_block(z, size=7, rate=r, block=i)
            skip += s

    # final logit layers
    with tf.sg_context(name='logit'):
        logit = (skip
                 .sg_conv1d(size=1, act='tanh', bn=True, name='conv_1')
                 .sg_conv1d(size=1, dim=voca_size, name='conv_2'))

return logit