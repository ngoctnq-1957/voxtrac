import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import keras
from scipy.io.wavfile import read as wavread_orig
import numpy as np
import math
from scipy.signal import stft, istft

lstm = keras.layers.LSTM(1026, activation='hard_sigmoid', recurrent_activation='hard_sigmoid',
                         use_bias=True, kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal', bias_initializer='zeros',
                         unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None,
                         bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         recurrent_constraint=None, bias_constraint=None,
                         dropout=0.0, recurrent_dropout=0.0, implementation=1,
                         return_sequences=True, return_state=False, go_backwards=False,
                         stateful=False, unroll=True)

blocksize = 1024
overlap = 768  # 3/4
sample_rate = 44100

def pcm2stft(pcm: 'wavread') -> {np.ndarray: 'powers', np.ndarray: 'phases'}:
    '''
    takes in a wavfile in matrix form,
    returns power and phase matrices.
    '''
    def stft_help(data):
        _, _, zxx = stft(data, window='hann', nperseg=blocksize,
                         noverlap=overlap, nfft=blocksize, return_onesided=True,
                         padded=False, axis=-1)
        return zxx
    # manual pad the pcm
    pad_size = math.ceil(pcm.shape[0]/256)*256 - pcm.shape[0]
    pcm = np.pad(pcm, ((pad_size//2, pad_size-pad_size//2), (0,0)), 'reflect')
    # start processing
    if len(pcm.shape) == 1:
        pcm = pcm[:,np.newaxis]
    powers = []
    phases = []
    for i in range(pcm.shape[1]):
        stft_temp = stft_help(pcm[:,i])
        # using magnitudes and angles
        powers.append(np.abs(stft_temp))
        phases.append(np.angle(stft_temp))
    return np.dstack(powers), np.dstack(phases)

def stft2pcm(powers: np.ndarray, phases: np.ndarray) -> 'wavread':
    '''
    takes in the magnitude and angle matrices,
    return the wavfile in matrix form.
    '''
    def istft_help(data):
        _, zxx = istft(data, window='hann', nperseg=blocksize,
                        noverlap=overlap, nfft=blocksize, input_onesided=True)
        return zxx
    acc = []
    for i in range(powers.shape[2]):
        acc.append(istft_help(powers[:,:,i] * np.exp(1j * phases[:,:,i])))
    return np.vstack(acc).T

def wavread(fname):
    rate, data = wavread_orig(fname)
    if data.dtype not in [np.float32, np.float64]:
        data = data / (np.iinfo(data.dtype).max + 1)
    # don't go vanishing on me
    return rate, data.astype(np.float32)

# path = '/content/drive/My Drive/Colab Notebooks/'
path = 'eval/'
rate, master_data = wavread(path+'mixed.wav')
master_data = np.vstack([master_data, master_data]).T
rate, vocal_data = wavread(path+'vocal.wav')
vocal_data = np.vstack([vocal_data, vocal_data]).T

master_pow, master_pha = pcm2stft(master_data)
vocal_pow, vocal_pha = pcm2stft(vocal_data)

master_pow = np.vstack([master_pow[:,:,0], master_pow[:,:,1]])
vocal_pow = np.vstack([vocal_pow[:,:,0], vocal_pow[:,:,1]])

master_pow = master_pow[:,:513]
vocal_pow = vocal_pow[:,:513]
true_mask = vocal_pow / master_pow

main_input = keras.Input(shape=(513,513*2), dtype='float32', name='audio_input')
x = lstm(main_input)
main_output = keras.layers.Multiply()([main_input, x])
model = keras.models.Model(inputs=main_input, outputs=main_output)
model.compile('adam', loss='mae')

model.fit(master_pow.T[np.newaxis,:,:], true_mask.T[np.newaxis,:,:], epochs=10)
