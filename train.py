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
            ## L: Text. (B, N), int32
            ## mels: Reduced melspectrogram. (B, T/r, n_mels) float32
            ## dones: (B, T/r) int32
            ## mags: Magnitude. (B, T, n_fft//2+1) float32
            if training:
                self.fnames, self.L, self.mels, self.dones, self.gts, self.mags, self.num_batch = get_batch()
                self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
            else:  # Inference
                self.L = tf.placeholder(tf.int32, shape=(hp.B, hp.max_N))
                self.mels = tf.placeholder(tf.float32, shape=(hp.B, hp.max_T // hp.r, hp.n_mels))
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
                        self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                                 mononotic_attention=(not training),
                                                                                 prev_max_attentions=self.prev_max_attentions)
                    with tf.variable_scope("AudioDec"):
                        # self.Y_logits, self.Y,  self.done_logits = AudioDec(self.R, training=training) # (B, T/r, n_mels)
                        self.Y_logits, self.Y, self.done_logits = AudioDec(self.R, training=training) # (B, T/r, n_mels)
                        if training:
                            self.Y_masks = tf.to_float(tf.expand_dims(self.dones, -1))
                        else:
                            self.Y_masks = tf.to_float(tf.expand_dims(tf.argmax(self.done_logits, -1), -1))
                        self.Y *= self.Y_masks
            else:  # num==2 & training
                with tf.variable_scope("SSRN"):
                    self.Z_logits, self.Z = SSRN(self.mels, training=training)
                    self.Z_masks = tf.to_float(tf.expand_dims(self.dones, -1))
                    self.Z_masks = tf.reshape(tf.tile(self.Z_masks, [1, 1, hp.r]), (hp.B, -1, 1))
                    self.Z *= self.Z_masks
            if not training:
                with tf.variable_scope("SSRN"):
                    self.Z_logits, self.Z = SSRN(self.Y, training=training)
                    # self.Z_masks = tf.to_float(tf.expand_dims(tf.argmax(self.done_logits, -1), -1))
                    # # print(self.Z_masks)
                    self.Z_masks = tf.reshape(tf.tile(self.Y_masks, [1, 1, hp.r]), (hp.B, -1, 1))
                    # # print(self.Z_masks)
                    # # print(self.Z)
                    self.Z *= self.Z_masks

            with tf.variable_scope("gs"):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if training:
                if num == 1:
                    # Loss
                    self.loss_mels = tf.reduce_sum(tf.abs(self.Y - self.mels))
                    self.loss_mels /= tf.reduce_sum(self.Y_masks*hp.n_mels)

                    self.loss_dones = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.done_logits, labels=self.dones))

                    self.bd = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits, labels=self.mels)
                    self.bd *= self.Y_masks
                    self.bd = tf.reduce_sum(self.bd) / (tf.reduce_sum(self.Y_masks)*hp.n_mels)

                    # A: (B, T/r, N)
                    # guided_attention: (B, hp.T/hp.r, hp.N)
                    self.A = tf.pad(self.alignments, [(0, 0), (0, hp.max_N), (0, hp.max_T//hp.r)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T//hp.r]
                    self._gts = self.gts
                    # self.gts = tf.pad(self.gts, [(0, 0), (0, hp.max_N), (0, hp.max_T//hp.r)], mode="CONSTANT", constant_values=0.)[:, :hp.max_N, :hp.max_T//hp.r]
                    self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
                    self.loss_att = tf.reduce_sum(tf.abs(self.A * self.gts) * self.attention_masks)
                    self.loss_att /= tf.reduce_sum(self.attention_masks)

                    self.loss = self.loss_mels + 0.1*self.bd + self.loss_att + self.loss_dones

                    tf.summary.scalar('Train_Loss/mels', self.loss_mels)
                    tf.summary.scalar('Train_Loss/bd', self.bd)
                    tf.summary.scalar('Train_Loss/dones', self.loss_dones)
                    tf.summary.scalar('Train_Loss/att', self.loss_att)
                    tf.summary.scalar('Train_Loss/LOSS', self.loss)
                    tf.summary.image('Train/mel_gt', tf.expand_dims(tf.transpose(self.mels[:1], [0,2,1]), -1))
                    tf.summary.image('Train/mel_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0,2,1]), -1))
                    tf.summary.text('Train/fname', self.fnames[0])
                else:
                    self.loss_mags = tf.reduce_sum(tf.abs(self.Z - self.mags))
                    self.loss_mags /= tf.to_float(tf.reduce_sum(self.Z_masks)*(1+hp.n_fft//2))

                    self.bd = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits, labels=self.mags)
                    self.bd *= self.Z_masks
                    self.bd = tf.reduce_sum(self.bd) / (tf.reduce_sum(self.Z_masks)*(1+hp.n_fft//2))
                    self.loss = self.loss_mags + self.bd

                    tf.summary.scalar('Train_Loss/mags', self.loss_mags)
                    tf.summary.scalar('Train_Loss/bd', self.bd)
                    tf.summary.scalar('Train_Loss/LOSS', self.loss)

                    tf.summary.image('Train/mag_gt', tf.expand_dims(tf.transpose(self.mags[:1], [0, 2, 1]), -1))
                    tf.summary.image('Train/mag_hat', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))

                # Training Scheme
                self.lr = learning_rate_decay(hp.lr, self.global_step)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                            beta1=hp.beta1,
                                                            beta2=hp.beta2,
                                                            epsilon=hp.eps)
                tf.summary.scalar("lr", self.lr)

                ## gradient clipping
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.clipped = []
                for grad, var in self.gvs:
                    # print(grad, var)
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
        sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)
        with sv.managed_session() as sess:
            # if num==1:
            #     # plot initial alignments
            #     alignments = sess.run(g.alignments)
            #     plot_alignment(alignments, 0, logdir)  # (Tx, Ty/r)

            while 1:
                for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    gs, _ = sess.run([g.global_step, g.train_op])

                    # Write checkpoint files at every 1k steps
                    if gs % 1000 == 0:
                        # fnames, mels, Y = sess.run([g.fnames, g.mels, g.Y])
                        # print("fname=", fnames[-1])
                        #
                        # np.save('mel_gt.npy', mels[-1])
                        # np.save('mel_hat.npy', Y[-1])
                        # Z = sess.run(g.mels)
                        # np.save("Z.npy", Z)
                        # wav = spectrogram2wav(Z[-1])
                        # from scipy.io.wavfile import write
                        # write("{}.wav".format("mag_wav"), hp.sr, wav)

                        sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))
                        if num == 1:
                            # plot alignments
                            alignments, gts = sess.run([g.alignments, g._gts])
                            plot_alignment(alignments[0], str(gs // 1000).zfill(3) + "k", logdir)  # (Tx, Ty)

                            # dones, done_logits = sess.run([g.dones, g.done_logits])
                            # print("done=", dones[0])
                            # # done = np.sum(dones[0])
                            # # print(done_logits.shape)
                            # print("done_logits=", np.argmax(done_logits, -1)[0])
                            # print(np.sum(np.argmax(done_logits, -1)[0]).shape)
                            # done_logit = np.sum(np.argmax(done_logits, -1)[0])
                            # print("_done=", done)
                            # print("_done_logits=", done_logit)
                # break
                if gs > hp.num_iterations: break

    print("Done")
