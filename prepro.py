# -*- coding: utf-8 -*-
# #/usr/bin/python2

'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

import numpy as np
import librosa

from hyperparams import Hyperparams as hp
import glob
import os
import codecs
from tqdm import tqdm
from utils import get_spectrograms, guided_attention

transcript = os.path.join("/data/private/voice", hp.data, 'transcript.csv')
mel_folder = os.path.join(hp.data, 'mels')
mag_folder = os.path.join(hp.data, 'mags')
gt_folder = os.path.join(hp.data, 'gts')

for folder in (mel_folder, mag_folder, gt_folder):
    if not os.path.exists(folder): os.mkdir(folder)

for line in tqdm(codecs.open(transcript, 'r', 'utf-8').readlines()):
    fpath, _, sent = line.strip().split("|")[:3]
    wav = os.path.join("/data/private/voice", hp.data, fpath)
    fname = os.path.basename(fpath).replace(".wav", ".npy")

    # features
    mel, mag = get_spectrograms(wav)  # (t, n_mels), (t, 1+n_fft/2) float32
    np.save(os.path.join(mel_folder, fname), mel)
    np.save(os.path.join(mag_folder, fname), mag)

    ## guided attention
    n = len(sent)
    t = len(mel)//hp.r
    gt = guided_attention(n, t)
    np.save(os.path.join(gt_folder, fname), gt)