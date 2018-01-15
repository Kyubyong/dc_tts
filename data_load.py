# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
import glob
from num2words import num2words
import librosa

# def text_normalize(sent):
#     '''Minimum text preprocessing'''
#     def _strip_accents(s):
#         return ''.join(c for c in unicodedata.normalize('NFD', s)
#                        if unicodedata.category(c) != 'Mn')
#
#     # sentence level
#     sent = _strip_accents(sent.lower())
#     sent = re.sub(u"[-—-]", " ", sent)
#     sent = re.sub("[^ a-z.?\d]", "", sent)
#
#     # word level
#     normalized = []
#     for word in sent.split():
#         srch = re.match("\d[\d,.]*$", word)
#         if srch:
#             word = num2words(float(word.replace(",", "")))
#         abb2exp = {'mr.':'mister', 'mrs': "misess", "dr.":"doctor", "no.": "number", "st.": "saint", "rev.": "reverend", "etc.":"et cetera"}
#         word = abb2exp.get(word, word)
#         normalized.append(word)
#     normalized = " ".join(normalized)
#     normalized = normalized.replace(".", " ")
#     normalized = re.sub("[ ]{2,}", " ", normalized)
#     normalized = normalized.strip()
#
#     return normalized

def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_data(training=True):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    fnames, ns, texts, mels, gts, mags = [], [], [], [], [], []
    num_samples = 1
    transcript = os.path.join("/data/private/voice", hp.data, 'transcript.csv')
    for line in codecs.open(transcript, 'r', 'utf-8'):
        fname, _, sent, isspoken = line.strip().split("|")
        # fname += ".wav"

        # Filtering
        # if isspoken=="1": continue
        if len(sent) > hp.max_N: continue
        duration = librosa.get_duration(filename=os.path.join("/data/private/voice", hp.data, fname))
        if duration > hp.max_duration: continue

        fname = os.path.basename(fname).replace(".wav", ".npy")
        sent += "E"  # E: EOS

        fnames.append(fname)
        ns.append(len(sent))
        texts.append(np.array([char2idx[char] for char in sent if char in char2idx], np.int32).tostring())
        mels.append(os.path.join(hp.data, "mels", fname))
        gts.append(os.path.join(hp.data, "gts", fname))
        mags.append(os.path.join(hp.data, "mags", fname))

        if num_samples==hp.B:
            if training: fnames, ns, texts, mels, gts, mags = [], [], [], [], [], []
            else: # for evaluation
                return fnames, ns, texts, mels, gts, mags
        num_samples += 1

    return fnames, ns, texts, mels, gts, mags

# def load_test_data():
#     # Load vocabulary
#     char2idx, idx2char = load_vocab()
#
#     # Parse
#     texts = []
#     for line in codecs.open('test_sents.txt', 'r', 'utf-8'):
#         sent = text_normalize(line).strip() + "E" # text normalization, E: EOS
#         if len(sent) <= hp.N:
#             sent += "P"*(hp.N-len(sent))
#             texts.append([char2idx[char] for char in sent])
#     return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fnames, ns, texts, mels, gts, mags = load_data()
        maxlen, minlen = max(ns), min(ns)

        # Calc total batch count
        num_batch = len(texts) // hp.B
         
        # Convert to string tensor
        fnames = tf.convert_to_tensor(fnames)
        ns = tf.convert_to_tensor(ns)
        texts = tf.convert_to_tensor(texts)
        mels = tf.convert_to_tensor(mels)
        gts = tf.convert_to_tensor(gts)
        mags = tf.convert_to_tensor(mags)

        # Create Queues
        fname, n, text, mel, gt, mag = tf.train.slice_input_producer([fnames, ns, texts, mels, gts, mags], shuffle=True)

        # Decoding
        text = tf.decode_raw(text, tf.int32) # (None,)

        def _load(mel, mag):
            x, y = np.load(mel), np.load(mag)
            t = x.shape[0]
            num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
            x = np.pad(x, [[0, num_paddings], [0, 0]], mode="constant")
            y = np.pad(y, [[0, num_paddings], [0, 0]], mode="constant")
            x = x[::hp.r, :]  # (t/r, n_mels) <- reduction
            return x, y

        mel, mag = tf.py_func(_load, [mel, mag], [tf.float32, tf.float32]) # (None, n_mels)
        gt = tf.py_func(lambda x:np.load(x), [gt], tf.float32) # (max_T, max_N)

        # Get done flag
        done = tf.ones_like(mel[:, 0], dtype=tf.int32)

        # Shape information
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft // 2 + 1))
        done.set_shape((None,))
        gt.set_shape((hp.max_N, hp.max_T//hp.r))

        # Batching
        _, (texts, mels, mags, dones, gts, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                        input_length=n,
                                        tensors=[text, mel, mag, done, gt, fname],
                                        batch_size=hp.B,
                                        bucket_boundaries=[i for i in range(minlen+1, maxlen-1, 20)],
                                        num_threads=32,
                                        capacity=hp.B * 4,
                                        dynamic_pad=True)

    return fnames, texts, mels, dones, gts, mags, num_batch