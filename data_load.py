# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

from hparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
import glob

def speaker2id(speaker):
    func = {speaker:id for id, speaker in enumerate(hp.speakers)}
    return func.get(speaker, None)

def id2speaker(id):
    func = {id:speaker for id, speaker in enumerate(hp.speakers)}
    return func.get(id, None)

def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    if mode=="train":
        files = glob.glob(hp.data)
        speaker_ids = [speaker2id(os.path.basename(f)[:4]) for f in files]
    else: # evaluation. samples.
        files = ("/data/public/rw/datasets/VCTK-Corpus/wav48/p253/p253_410.wav")
        speaker_ids = [speaker2id("p253")]
    return files, speaker_ids

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        files, speakers = load_data() # list

        # Calc total batch count
        num_batch = len(files) // hp.batch_size

        # Create Queues
        f, speaker = tf.train.slice_input_producer([files, speakers], shuffle=True)

        # Parse
        wav, qt = tf.py_func(get_wav, [f], [tf.float32, tf.int32])  # (T, 1)

        # Add shape information
        wav.set_shape((hp.length, 1))
        qt.set_shape((hp.length, 1))

        # Batching
        speakers, wavs, qts = tf.train.batch(tensors=[speaker, wav, qt],
                                                        batch_size=hp.batch_size,
                                                        num_threads=32)

    return speakers, wavs, qts, num_batch