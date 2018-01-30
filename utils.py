
from hparams import Hyperparams as hp
import librosa
import numpy as np

def mu_law_encode(audio):
    '''Quantizes waveform amplitudes.
    Mostly adaped from
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py#L64-L75

    Args:
      audio: Raw wave signal. float32.
    '''
    mu = float(hp.quantization_channels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = min(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)


def mu_law_decode(output):
    '''Recovers waveform from quantized values.
    Mostly adapted from
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py#L64-L75
    '''
    mu = hp.quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (output.astype(np.float32) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return np.sign(signal) * magnitude

def get_wav(fpath):
    wav, sr = librosa.load(fpath, sr=hp.sr)
    wav, _ = librosa.effects.trim(wav)
    wav /= np.abs(wav).max()
    qt = mu_law_encode(wav)

    # Padding
    maxlen = hp.duration * hp.sr
    wav = np.pad(wav, ([0, maxlen]), mode="constant")[:maxlen]
    qt = np.pad(qt, ([0, maxlen]), mode="constant")[:maxlen]

    ####
    wav = np.expand_dims(wav, -1) # (T, 1)
    qt = np.expand_dims(qt, -1) # (T, 1)

    return wav[:-1, :], qt[1:, :] # why?
