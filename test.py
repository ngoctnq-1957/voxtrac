from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.signal import stft, istft
import numpy as np

rate, data = wavread('kwkt_instrumentals/hey_kids.wav')
print(data.shape)
_, _, zxx1 = stft(data[:,0])
_, _, zxx2 = stft(data[:,1])
zxx1_pow = np.abs(zxx1)
zxx2_pow = np.abs(zxx2)

zxx1_phase = np.angle(zxx1)
zxx2_phase = np.angle(zxx2)

mask = np.zeros(zxx1_pow.shape)
mask[:10,:] = 1
mask[50:80,:] = 1

new_zxx1 = mask * zxx1_pow * np.exp(1j * zxx1_phase)
new_zxx2 = mask * zxx2_pow * np.exp(1j * zxx2_phase)

data21 = istft(new_zxx1)[1][:,np.newaxis]
data22 = istft(new_zxx2)[1][:,np.newaxis]
print(data21.shape, data22.shape)
data2 = np.hstack([data21,data22])
print(data2.shape)
wavwrite('test.wav', rate, data2)