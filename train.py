# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import *
import sys


class Graph:
    def __init__(self, num=1, training=True):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Data Feeding
            ## x: Text. (B, N), int32
            ## mels: Reduced melspectrogram. (B, T/r, n_mels) float32
            ## mag: Magnitude. (B, T, n_fft//2+1) float32
            if training:
                self.L, self.mels, self.mags, self.num_batch = get_batch()
                self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
            else:  # Inference
                self.L = tf.placeholder(tf.int32, shape=(hp.B, hp.N))
                self.mels = tf.placeholder(tf.float32, shape=(hp.B, hp.T // hp.r, hp.n_mels))
                self.prev_max_attentions = tf.placeholder(tf.int32, shape=(hp.B,))

            if num == 1 or (not training):
                with tf.variable_scope("Text2Mel"):
                    # Get S or decoder inputs. (B, T//r, n_mels)
                    self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

                    # Networks
                    with tf.variable_scope("TextEnc"):
                        self.K, self.V = TextEnc(self.L, training=training)  # (N, Tx, e)

                    with tf.variable_scope("AudioEnc"):
                        self.Q = AudioEnc(self.S, training=training)

                    with tf.variable_scope("Attention"):
                        # R: (B, T/r, 2d)
                        # A: (B, T/r, N)
                        # alignments: (N, T/r)
                        # max_attentions: (B, T/r)
                        self.R, self.A, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                                 mononotic_attention=(not training),
                                                                          prev_max_attentions=self.prev_max_attentions)
                    with tf.variable_scope("AudioDec"):
                        self.logits, self.Y = AudioDec(self.R, training=training) # (B, T/r, n_mels)
            else:  # num==2 & training
                with tf.variable_scope("SSRN"):
                    self.logits, self.Z = SSRN(self.mels, training=training)

            if not training:
                with tf.variable_scope("SSRN"):
                    self.logits, self.Z = SSRN(self.Y, training=training)

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if training:
                if num == 1:
                    # Loss
                    self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))
                    self.bd = tf.reduce_mean(tf.abs(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.mels)))
                    self.loss_att = tf.reduce_mean(tf.abs(self.A * guided_attention()))
                    self.loss = 0.25*(self.loss_mels + self.bd) + 0.5*self.loss_att

                    tf.summary.scalar('Train_Loss/mels', self.loss_mels)
                    tf.summary.scalar('Train_Loss/bd', self.bd)
                    tf.summary.scalar('Train_Loss/att', self.loss_att)
                    tf.summary.scalar('Train_Loss/LOSS', self.loss)
                else:
                    self.loss_mags = tf.reduce_mean(tf.abs(self.Z - self.mags))
                    self.bd = tf.reduce_mean(tf.abs(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.mags)))
                    self.loss = 0.5*(self.loss_mags + self.bd)

                    tf.summary.scalar('Train_Loss/mags', self.loss_mags)
                    tf.summary.scalar('Train_Loss/bd', self.bd)
                    tf.summary.scalar('Train_Loss/LOSS', self.loss)

                # Training Scheme
                self.lr = learning_rate_decay(hp.lr, self.global_step)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                        beta1=hp.beta1,
                                                        beta2=hp.beta2,
                                                        epsilon=hp.eps)
                ## gradient clipping
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.clipped = []
                for grad, var in self.gvs:
                    grad = tf.clip_by_value(grad, -1. * hp.max_grad_val, hp.max_grad_val)
                    grad = tf.clip_by_norm(grad, hp.max_grad_norm)
                    self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    num = int(sys.argv[1])

    g = Graph(num=num); print("Training Graph loaded")
    logdir = hp.logdir + "-" + str(num)
    with g.graph.as_default():
        sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0)
        with sv.managed_session() as sess:
            if num==1:
                # plot initial alignments
                alignments = sess.run(g.alignments)
                plot_alignment(alignments, 0, logdir)  # (Tx, Ty/r)

            while 1:
                if sv.should_stop(): break
                for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    gs, _ = sess.run([g.global_step, g.train_op])

                    # Write checkpoint files at every 1k steps
                    if gs % 1000 == 0:
                        sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                        if num == 1:
                            # plot alignments
                            alignments = sess.run(g.alignments)
                            plot_alignment(alignments, str(gs // 1000).zfill(3) + "k", logdir)  # (Tx, Ty)

                # break
                if gs > hp.num_iterations: break

    print("Done")
