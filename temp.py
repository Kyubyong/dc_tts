import tensorflow as tf
from modules import *
from hparams import Hyperparams as hp
# def encoder(inputs):
#     for i in range(hp.encoder_layers):
#         inputs = conv1d(inputs,
#                        filters=hp.d,
#                        size=hp.winsize,
#                        strides=hp.stride,
#                        padding="valid",
#                        activation_fn=tf.nn.relu if i < hp.encoder_layers-1 else None,
#                         scope="conv1d_{}".format(i))
#     z = inputs
#     return z
# tf.norm
# inputs = tf.ones((1, 16000, 1))
# z = encoder(inputs)
# print(z)
a = tf.ones((1, 10, 20, 30))
b = tf.ones((1, 1, 20, 30))
c = tf.norm(a-b, axis=-1)
k = tf.argmin(c, axis=-1)

embed = tf.ones((10, 40), tf.float32)
k = tf.ones((2, 10), tf.int32)
out = tf.gather(embed, k)
print(out)