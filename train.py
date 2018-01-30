# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import encoder, vq,
import tensorflow as tf
from utils import *
import sys


class Graph:
    def __init__(self, mode="train"):
        '''
        Args:
          mode: Either "train" or "generate".
        '''
        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## x: Raw wav. (B, length), float32
        ## qt: Quantized wav. (B, length) int32
        ## speaker: Speaker id. [0, 108]. int32.
        if mode=="train":
            self.x, self.qt, self.speaker, self.num_batch = get_batch()
        else:  # Synthesize
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))

        # encoder
        self.z_e = encoder(self.x) # (B, T', D)

        # vq
        self.z_q = vq(self.z_e) # (B, T', D)

        # decoder
        self.x_rec = decoder(self.z_q)

        if training:
            #
            self.loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.x_logits, labels=self.qt))
            self.loss2 = tf.reduce_mean(tf.squared_difference(sg(self.z), self.e))
            self.loss3 = hp.beta * tf.reduce_mean(tf.squared_difference(self.z, sg(self.e)))
            self.loss = self.loss1 + self.loss2 + self.loss3


            tf.summary.scalar('train/loss1', self.loss1)
            tf.summary.scalar('train/loss2', self.loss2)
            tf.summary.scalar('train/loss3', self.loss3)
            tf.summary.scalar('train/LOSS', self.loss)

            # Training Scheme
            self.lr = learning_rate_decay(hp.lr, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tf.summary.scalar("lr", self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_value(grad, -1., 1.)
                self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # argument: 1 or 2. 1 for Text2mel, 2 for SSRN.
    num = int(sys.argv[1])

    g = Graph(num=num); print("Training Graph loaded")

    logdir = hp.logdir + "-" + str(num)
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)
    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                    if num==1:
                        # plot alignment
                        alignments = sess.run(g.alignments)
                        plot_alignment(alignments[0], str(gs // 1000).zfill(3) + "k", logdir)

                # break
                if gs > hp.num_iterations: break

print("Done")



z = encoder(x)
e = vq(z)



...
y = dec(qt, cond)

loss1 = softmax_ce(y, t)
loss2 = F.squared_difference(z, e_)
loss3 = self.beta * F.mean((z - e)) ** 2)
loss = loss1 + loss2 + loss3