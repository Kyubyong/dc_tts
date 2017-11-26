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
import tqdm


def get_spectrograms(sound_file):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)
    if len(linear)!=513: print(linear.shape)
    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = librosa.amplitude_to_db(mel)
    mag = librosa.amplitude_to_db(mag)

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

if __name__ == "__main__":
    wav_folder = os.path.join('/data/private/voice/LJSpeech-1.0', 'wavs')
    # wav_folder = os.path.join('/data/private/voice/', 'Tom')
    mel_folder = os.path.join(hp.data, 'mels')
    mag_folder = os.path.join(hp.data, 'mags')

    for folder in (mel_folder, mag_folder):
        if not os.path.exists(folder): os.mkdir(folder)

    files = glob.glob(os.path.join(wav_folder, "*"))
    for f in tqdm.tqdm(files):
        fname = os.path.basename(f)
        mel, mag = get_spectrograms(f)  # (n_mels, T), (1+n_fft/2, T) float32
        np.save(os.path.join(mel_folder, fname.replace(".wav", ".npy")), mel)
        np.save(os.path.join(mag_folder, fname.replace(".wav", ".npy")), mag)
