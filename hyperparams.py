# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''
import math

def get_max_T(max_duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(max_duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T

class Hyperparams:
    '''Hyper parameters'''
    # signal processing
    sr = 22050 # Sampling rate.
    n_fft = 1024 # fft points (samples)
    hop_length = 256 # samples  This is dependent on the frame_shift.
    win_length = 1024 # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # Model
    r = 4 # Reduction factor. Do not change this.
    dropout_rate = 0.1
    vocab_size = 32 # [PE a-z'.?]
    e = 128 # == embedding
    d = 256
    c = 512 # == hidden units of SSRN
    attention_win_size = 3

    # data
    data = 'nick' #'LJSpeech-1.0'#### # or 'nick (internal)'
    max_duration = 15.0 # Maximum length of a sound file in seconds.
    max_N = 300 # Maximum number of characters.
    max_T = int(get_max_T(max_duration, sr, hop_length, r)) # Maximum number of frames

    # training scheme
    lr, beta1, beta2, eps = 0.002, 0.5, 0.99, 10e-6
    # logdir = "logdir/L107" #lr=0.001
    # logdir = "logdir/L108" #lr=0.0005
    logdir = "logdir/L110" #lr=0.002
    sampledir = 'samples/L104'
    B = 32 # batch size
    max_grad_val = 5
    max_grad_norm = 100
    num_iterations = 2000000
