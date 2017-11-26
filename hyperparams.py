# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''
import math

def get_T(duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T

class Hyperparams:
    '''Hyper parameters'''
    # signal processing
    sr = 22050 # Sampling rate.
    n_fft = 1024 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
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
    dropout_rate = .05

    ## Enocder
    vocab_size = 32 # [PE a-z'.?]
    e = 128 # == embedding
    d = 256
    c = 512 # == hidden units of SSRN
    ## Decoder
    dec_layers = 4
    dec_filter_size = 5
    attention_size = 128*2 # == a
    ## Converter
    converter_layers = 5*2
    converter_filter_size = 5
    converter_channels = 256 # == v

    sinusoid = False
    attention_win_size = 3

    max_grad_val = 5
    max_grad_norm = 100

    # data
    data = 'LJSpeech-1.0'#'nick'### # or 'nick (internal)'
    max_duration = 10.0 # Maximum length of a sound file in seconds.
    N = 180 # Maximum number of characters.
    T = int(get_T(max_duration, sr, hop_length, r)) # Maximum number of frames

    # training scheme
    lr, beta1, beta2, eps = 0.0005, 0.5, 0.9, 10e-6
    logdir = "logdir/L03"
    sampledir = 'samples/L03'
    B = 16 # batch size
    num_iterations = 1000000

