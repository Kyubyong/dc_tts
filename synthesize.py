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
from data_load import load_test_data
from scipy.io.wavfile import write

def synthesize():
    # Load data
    X = load_test_data()

    # Load graph
    g = Graph(training=False); print("Graph loaded")

    # Inference
    with g.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()

            # Restore parameters
            saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]

            # Synthesize
            file_id = 1
            for i in range(0, len(X), hp.batch_size):
                x = X[i:i + hp.batch_size]

                # Get melspectrogram
                mel_output = np.zeros((hp.batch_size, hp.Ty // hp.r, hp.n_mels * hp.r), np.float32)
                decoder_output = np.zeros((hp.batch_size, hp.Ty // hp.r, hp.embed_size), np.float32)
                alignments_li = np.zeros((hp.dec_layers, hp.Tx, hp.Ty//hp.r), np.float32)
                prev_max_attentions_li = np.zeros((hp.dec_layers, hp.batch_size), np.int32)
                for j in range(hp.Ty // hp.r):
                    _gs, _mel_output, _decoder_output, _max_attentions_li, _alignments_li = \
                        sess.run([g.global_step, g.mel_output, g.decoder_output, g.max_attentions_li, g.alignments_li],
                                 {g.x: x,
                                  g.y1: mel_output,
                                  g.prev_max_attentions_li:prev_max_attentions_li})
                    mel_output[:, j, :] = _mel_output[:, j, :]
                    decoder_output[:, j, :] = _decoder_output[:, j, :]
                    alignments_li[:, :, j] = np.array(_alignments_li)[:, :, j]
                    prev_max_attentions_li = np.array(_max_attentions_li)[:, :, j]

                # Get magnitude
                mag_output = sess.run(g.mag_output, {g.decoder_output: decoder_output})

                # Generate wav files
                if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
                for mag in mag_output:
                    print("Working on file num ", file_id)
                    wav = spectrogram2wav(mag)
                    write(hp.sampledir + "/{}_{}.wav".format(mname, file_id), hp.sr, wav)
                    file_id += 1

if __name__ == '__main__':
    synthesize()
    print("Done")


