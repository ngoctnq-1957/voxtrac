import numpy as np
import os
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.signal import stft, istft

#== HYPERPARAMETERS ==#
SEG_CONST = 512

#== DEFAULT VALUES ==#
sample_rate = 44100

#== TEST VARIABLES ==#
mat_before = None
mat_after = None

def load_audio(audio_path):
    global sample_rate
    sample_rate, data = wavread(audio_path)

    # why normalize?
    # data /= (2 ** 10)

    channels = []

    global mat_before
    mat_before = data
    
    for i in range(data.shape[1]):
        freqs, bins, sxx = stft(data[:,i], nperseg=SEG_CONST, fs=sample_rate)
        real_part = np.real(sxx)
        real_part = np.expand_dims(real_part, axis=-1)
        complex_part = np.imag(sxx)
        complex_part = np.expand_dims(complex_part, axis=-1)
        channels.append(np.concatenate((real_part, complex_part), axis=-1)[:,:,:,np.newaxis])

    ret = np.concatenate(channels, axis=-1)
    return ret

def save_audio(data, audio_path):
    channels = []
    for i in range(data.shape[-1]):
        # data *= (2 ** 10)
        audio = data[:, :, 0, i] + data[:, :, 1, i] * 1j
        _, x = istft(np.abs(audio) * np.exp(1j * np.angle(audio)), fs=sample_rate, nperseg=SEG_CONST)
        # x = x.astype(np.int16)
        channels.append(np.expand_dims(x, axis=-1))

    ret = np.concatenate(channels, axis=-1)
    wavwrite(audio_path, sample_rate, ret)
    return ret

if __name__ == "__main__":
    data = load_audio('load.wav')
    ret = save_audio(data, 'save.wav')

    print(np.linalg.norm(ret-mat_before))
