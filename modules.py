# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor. 
    
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    return outputs


def normalize(inputs,
              type="bn",
              epsilon=1e-8,
              decay=0.999,
              training=True,
              reuse=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension. Or if type is `ln`, the normalization is over
        the last dimension.
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average.
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type == "bn":
        # param_dim = inputs.get_shape()[-1:]
        # scale = tf.Variable(tf.ones([param_dim]))
        # beta = tf.Variable(tf.zeros([param_dim]))
        # pop_mean = tf.Variable(tf.zeros([param_dim]), trainable=False)
        # pop_var = tf.Variable(tf.ones([param_dim]), trainable=False)
        #
        # if training:
        #     batch_mean, batch_var = tf.nn.moments(inputs, [0], keep_dims=True)
        #     print(inputs)
        #     print(batch_mean)
        #     print(batch_var)
        #     print(scale, beta, pop_mean, pop_var)
        #     train_mean = tf.assign(pop_mean,
        #                            pop_mean * decay + batch_mean * (1 - decay))
        #     train_var = tf.assign(pop_var,
        #                           pop_var * decay + batch_var * (1 - decay))
        #     with tf.control_dependencies([train_mean, train_var]):
        #         return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        # else:
        #     return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
        outputs = tf.layers.batch_normalization(inputs,
                                                training=training)
    elif type in ("ln", "ins"):
        if type=="ln":
            outputs = tf.contrib.layers.layer_norm(inputs,
                                                   begin_norm_axis=-1,
                                                   scope=scope)
    else:
        outputs = inputs

    return outputs


def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387

    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        outputs = H * T + inputs * (1. - T)
    return outputs

def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           dropout_rate=0,
           use_bias=True,
           norm_type=None,
           activation_fn=None,
           training=True,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
                  "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(), "reuse": reuse}

        tensor = tf.layers.conv1d(**params)
        tensor = normalize(tensor, type=norm_type, training=training)
        if activation_fn is not None:
            tensor = activation_fn(tensor)

        tensor \
            = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor

def hc(inputs,
       filters=None,
       size=1,
       rate=1,
       padding="SAME",
       dropout_rate=0,
       use_bias=True,
       norm_type=None,
       activation_fn=None,
       training=True,
       scope="hc",
       reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    _inputs = inputs
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]


        params = {"inputs": inputs, "filters": 2*filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
                  "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(), "reuse": reuse}

        tensor = tf.layers.conv1d(**params)
        H1, H2 = tf.split(tensor, 2, axis=-1)
        H1 = normalize(H1, type=norm_type, training=training, scope="H1")
        H2 = normalize(H2, type=norm_type, training=training, scope="H2")
        H1 = tf.nn.sigmoid(H1, "gate")
        H2 = activation_fn(H2, "info") if activation_fn is not None else H2
        tensor = H1*H2 + (1.-H1)*_inputs


        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor

def conv1d_transpose(inputs,
                     filters=None,
                     size=3,
                     stride=2,
                     padding='same',
                     dropout_rate=0,
                     norm_type=None,
                     activation=None,
                     training=True,
                     use_bias=True,
                     scope="conv1d_transpose",
                     reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        if filters is None:
            filters = inputs.get_shape().as_list()[-1]
        inputs = tf.expand_dims(inputs, 1)
        tensor = tf.layers.conv2d_transpose(inputs,
                                   filters=filters,
                                   kernel_size=(1, size),
                                   strides=(1, stride),
                                   padding=padding,
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   use_bias=use_bias)
        tensor = tf.squeeze(tensor, 1)
        tensor = normalize(tensor, type=norm_type, training=training)
        if activation is not None:
            tensor = activation(tensor)

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor





