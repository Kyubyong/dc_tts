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
from num2words import num2words

# def text_normalize(sent):
#     '''Minimum text preprocessing'''
#     def _strip_accents(s):
#         return ''.join(c for c in unicodedata.normalize('NFD', s)
#                        if unicodedata.category(c) != 'Mn')
#
#     normalized = []
#     for word in sent.split():
#         word = _strip_accents(word.lower())
#         srch = re.match("\d[\d,.]*$", word)
#         if srch:
#             word = num2words(float(word.replace(",", "")))
#         word = re.sub(u"[-—-]", " ", word)
#         word = re.sub("[^ a-z'.?]", "", word)
#         normalized.append(word)
#     normalized = " ".join(normalized)
#     normalized = re.sub("[ ]{2,}", " ", normalized)
#     normalized = normalized.strip()
#
#     return normalized

def text_normalize(sent):
    '''Minimum text preprocessing'''
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    # sentence level
    sent = _strip_accents(sent.lower())
    sent = re.sub(u"[-—-]", " ", sent)
    sent = re.sub("[^ a-z.?\d]", "", sent)

    # word level
    normalized = []
    for word in sent.split():
        srch = re.match("\d[\d,.]*$", word)
        if srch:
            word = num2words(float(word.replace(",", "")))
        abb2exp = {'mr.':'mister', 'mrs': "misess", "dr.":"doctor", "no.": "number", "st.": "saint", "rev.": "reverend", "etc.":"et cetera"}
        word = abb2exp.get(word, word)
        normalized.append(word)
    normalized = " ".join(normalized)
    normalized = normalized.replace(".", " ")
    normalized = re.sub("[ ]{2,}", " ", normalized)
    normalized = normalized.strip()

    return normalized

def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_data(training=True):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts, mels, mags = [], [], []
    num_samples = 1
    if hp.data == "LJSpeech-1.0":
        metadata = os.path.join(hp.data, 'metadata.csv')
        for line in codecs.open(metadata, 'r', 'utf-8'):
            fname, _, sent = line.strip().split("|")
            sent = text_normalize(sent) + "E" # text normalization, E: EOS
            if len(sent) <= hp.N:
                texts.append(np.array([char2idx[char] for char in sent], np.int32).tostring())
                mels.append(os.path.join(hp.data, "mels", fname + ".npy"))
                mags.append(os.path.join(hp.data, "mags", fname + ".npy"))

                if num_samples==hp.B:
                    if training: texts, mels, mags = [], [], []
                    else: # for evaluation
                        num_samples += 1
                        return texts, mels, mags
                num_samples += 1
    else: # nick
        metadata = os.path.join(hp.data, 'metadata.tsv')
        for line in codecs.open(metadata, 'r', 'utf-8'):
            fname, sent = line.strip().split("\t")
            sent = text_normalize(sent) + "E"  # text normalization, E: EOS
            if len(sent) <= hp.N:
                texts.append(np.array([char2idx[char] for char in sent], np.int32).tostring())
                mels.append(os.path.join(hp.data, "mels", fname.split("/")[-1].replace(".wav", ".npy")))
                mags.append(os.path.join(hp.data, "mags", fname.split("/")[-1].replace(".wav", ".npy")))

                if num_samples==hp.B:
                    if training: texts, mels, mags = [], [], []
                    else: # for evaluation
                        num_samples += 1
                        return texts, mels, mags
                num_samples += 1
    return texts, mels, mags

def load_test_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts = []
    for line in codecs.open('test_sents.txt', 'r', 'utf-8'):
        sent = text_normalize(line).strip() + "E" # text normalization, E: EOS
        if len(sent) <= hp.N:
            sent += "P"*(hp.N-len(sent))
            texts.append([char2idx[char] for char in sent])
    texts = np.array(texts, np.int32)
    return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        _texts, _mels, _mags = load_data()

        # Calc total batch count
        num_batch = len(_texts) // hp.B
         
        # Convert to string tensor
        texts = tf.convert_to_tensor(_texts)
        mels = tf.convert_to_tensor(_mels)
        mags = tf.convert_to_tensor(_mags)
         
        # Create Queues
        text, mel, mag = tf.train.slice_input_producer([texts, mels, mags], shuffle=True)

        # Decoding
        text = tf.decode_raw(text, tf.int32) # (None,)
        mel = tf.py_func(lambda x:np.load(x), [mel], tf.float32) # (None, n_mels)
        mag = tf.py_func(lambda x:np.load(x), [mag], tf.float32) # (None, 1+n_fft/2)

        # Padding
        text = tf.pad(text, ((0, hp.N),))[:hp.N] # (N,)
        mel = tf.pad(mel, ((0, hp.T), (0, 0)))[:hp.T] # (T, n_mels)
        mag = tf.pad(mag, ((0, hp.T), (0, 0)))[:hp.T] # (T, 1+n_fft/2)

        # Reduction
        mel = mel[::hp.r, :] # (T/r, n_mels)

        # create batch queues
        texts, mels, mags = tf.train.batch([text, mel, mag],
                                shapes=[(hp.N,), (hp.T//hp.r, hp.n_mels), (hp.T, 1+hp.n_fft//2)],
                                num_threads=32,
                                batch_size=hp.B, 
                                capacity=hp.B*32,   
                                dynamic_pad=False)

    return texts, mels, mags, num_batch