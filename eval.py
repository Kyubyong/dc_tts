# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data

def eval():
    # Load data
    lengths, texts, mels, mags = load_data(training=False)

    L = np.array([np.fromstring(text, np.int32) for text in texts])
    mels = np.array([np.load(mel) for mel in mels])
    mags = np.array([np.load(mag) for mag in mags])

    # Padding
    L = np.array([np.pad(each, ((0, hp.N),), "constant")[:hp.N] for each in L])
    mels = np.array([np.pad(each, ((0, hp.T), (0, 0)), "constant")[:hp.T] for each in mels])
    mags = np.array([np.pad(each, ((0, hp.T), (0, 0)), "constant")[:hp.T] for each in mags])

    # mel reduction
    mels = mels[:, ::hp.r, :]

    # Load graph
    g = Graph(training=False); print("Graph loaded")

    # Inference
    with g.graph.as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Restore parameters
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
            saver1 = tf.train.Saver(var_list=var_list)
            saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1")); print("Text2Mel Restored!")

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN')
            saver2 = tf.train.Saver(var_list=var_list)
            saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2")); print("SSRN Restored!")

            # Writer
            writer = tf.summary.FileWriter(hp.logdir, sess.graph)

            # Get melspectrogram
            Y = np.zeros((hp.B, hp.T // hp.r, hp.n_mels), np.float32)
            alignments = np.zeros((hp.N, hp.T //hp.r), np.float32)
            prev_max_attentions = np.zeros((hp.B,), np.int32)
            for j in range(hp.T // hp.r):
                _gs, _Y, _max_attentions, _alignments = \
                    sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                             {g.L: L,
                              g.mels: Y,
                              g.prev_max_attentions:prev_max_attentions})
                Y[:, j, :] = _Y[:, j, :]
                alignments[:, j] = _alignments[:, j]
                prev_max_attentions = _max_attentions[:, j]

            # Get magnitude
            Z = sess.run(g.Z, {g.Y: Y})

            # Loss
            eval_loss_mels = np.mean(np.abs(Y - mels))
            eval_loss_mags = np.mean(np.abs(Z - mags))
            eval_loss = eval_loss_mels + eval_loss_mags

            # Generate the first wav file
            sent = "".join(g.idx2char[xx] for xx in L[-1]).split("E")[0]
            wav = spectrogram2wav(Z[-1])
            wav = np.expand_dims(wav, 0)

            # Summary
            tf.summary.scalar("Eval_Loss/mels", eval_loss_mels)
            tf.summary.scalar("Eval_Loss/mags", eval_loss_mags)
            tf.summary.scalar("Eval_Loss/LOSS", eval_loss)
            tf.summary.text("Sent", tf.convert_to_tensor(sent))
            tf.summary.audio("audio sample", wav, hp.sr, max_outputs=1)
            tf.summary.image("alignments", np.expand_dims(np.expand_dims(alignments, 0), -1), max_outputs=len(alignments))

            merged = tf.summary.merge_all()
            writer.add_summary(sess.run(merged), global_step=_gs)
            writer.close()

if __name__ == '__main__':
    eval()
    print("Done")


