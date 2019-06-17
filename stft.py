import tensorflow as tf
import librosa
import scipy
from scipy.io.wavfile import read as wavread
import numpy as np

# load in audio as an array
data, sample_rate = librosa.load('load.wav', sr=None, mono=False)
_, data2 = wavread('load.wav', True)

print(np.max(np.abs(data2-data.T)))

# sample partially
data = data[:100]

# FFT window size to be power of 2, exactly nonoverlapping
chunk_size = 4

# try manual padding of synthesized data
data = np.array([2,6,0,8,1,9,9,5]).astype(np.float32)

# SciPy
_, _, scipy_stft = scipy.signal.stft(data, window='hann', nperseg=chunk_size,
        noverlap=chunk_size*3//4, nfft=chunk_size, return_onesided=True,
        padded=True, axis=-1)
_, scipy_istft = scipy.signal.istft(scipy_stft, fs=sample_rate, window='hann',
            nperseg=chunk_size, noverlap=chunk_size*3//4, nfft=chunk_size, input_onesided=True)

# librosa
rosa_stft = librosa.stft(data, n_fft=chunk_size, hop_length=chunk_size//4,
        win_length=chunk_size, window='hann', center=True,
        pad_mode='reflect')
rosa_istft = librosa.istft(rosa_stft, hop_length=chunk_size//4, win_length=chunk_size,
            window='hann', center=True)

# TensorFlow
tf.InteractiveSession()
tf_stft = tf.signal.stft(data, frame_length=chunk_size, frame_step=chunk_size//4,
        fft_length=chunk_size, window_fn=tf.signal.hann_window,
        pad_end=True, name=None)
tf_istft = tf.signal.inverse_stft(tf_stft, frame_length=chunk_size,
            frame_step=chunk_size//4, fft_length=chunk_size,
            window_fn=tf.signal.hann_window, name=None)
tf_stft = tf.transpose(tf_stft).eval()
tf_istft = tf_istft.eval()

print(scipy_stft.shape, rosa_stft.shape, tf_stft.shape)

print('scipy\n', scipy_stft)
print('rosa\n', rosa_stft)
print('tf\n', tf_stft)
print()
print('scipy\n', scipy_istft)
print('rosa\n', rosa_istft)
print('tf\n', tf_istft)