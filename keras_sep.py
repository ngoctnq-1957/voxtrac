#!/usr/bin/env python3
import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# %% imports
import keras
from keras.layers import Input, SeparableConv2D, LeakyReLU, Dropout, \
        UpSampling2D, Concatenate, Cropping2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import os
import random
import operator
import numpy as np
import math
from scipy.io.wavfile import read as wavread_orig
from scipy.io.wavfile import write as wavwrite
from scipy.signal import stft, istft

# %% Constants
blocksize = 1024
overlap = 768  # 3/4
sample_rate = 44100

# %% conversions between PCM and STFT
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
    pcm = np.pad(pcm, (pad_size, 0), 'reflect')
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
    if data.dtype != np.float32:
        data = data / (np.iinfo(data.dtype).max + 1)
    return rate, data

# %% building the model
relu_rate = .2
drop_rate = .1
# for now
channel = 1
stft_len = 513

main_input = Input(shape=(513,stft_len,channel), dtype='float32', name='audio_input')
# 2 convolutional layers, ends with a dropout
x = SeparableConv2D(filters=2, kernel_size=5)(main_input)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=4, kernel_size=5)(x)
x = LeakyReLU(alpha=relu_rate)(x)
skip = Dropout(rate=drop_rate)(x)

# Pooling
x = MaxPooling2D(pool_size=3, strides=2)(skip)

# 2 more convolutions
x = SeparableConv2D(filters=4, kernel_size=5)(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=4, kernel_size=5)(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = Dropout(rate=drop_rate)(x)

# upsample
x = UpSampling2D(size=2)(x)
# padding -- skip is larger than x
left_pad = (skip.shape[1] - x.shape[1]) // 2
right_pad = skip.shape[1] - x.shape[1] - left_pad
up_pad = (skip.shape[2] - x.shape[2]) // 2
down_pad = skip.shape[2] - x.shape[2] - up_pad
left_pad = int(left_pad)
right_pad = int(right_pad)
up_pad = int(up_pad)
down_pad = int(down_pad)

crop = Cropping2D(((left_pad, right_pad), (up_pad, down_pad)))(skip)
x = Concatenate(axis=-1)([x, crop])
# and convolve
x = SeparableConv2D(filters=2, kernel_size=5)(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=1, kernel_size=5)(x)
x = LeakyReLU(alpha=relu_rate)(x)

left_pad = (513 - x.shape[1]) // 2
right_pad = 513 - x.shape[1] - left_pad
up_pad = (stft_len - x.shape[2]) // 2
down_pad = stft_len - x.shape[2] - up_pad
left_pad = int(left_pad)
right_pad = int(right_pad)
up_pad = int(up_pad)
down_pad = int(down_pad)

main_output = ZeroPadding2D(((left_pad, right_pad), (up_pad, down_pad)))(x)
model = Model(inputs=main_input, outputs=main_output)

def get_lrate(epoch, lr):
    ''' learning rate
    drop = 0.5
    epochs_drop = 2.0
    lrate = lr * math.pow(drop, 
        math.floor((1+epoch)/epochs_drop))
    '''
    lrate = min(.1, 5/(epoch+1))
    return lrate

lr = LearningRateScheduler(get_lrate)
model.compile(optimizer=RMSprop(lr=.1), loss='mse')

stoppu = EarlyStopping(min_delta=1e-8, patience=5)
btfu = ModelCheckpoint('vox.hdf5', save_best_only=False, period=1)

# %% run.
data_dir = '/home/tran.ngo.quang.ngoc/Downloads/MIR-1K_for_MIREX/Wavfile/'
sample_rate, fmat = wavread(data_dir+'abjones_1_01.wav')
master = fmat[:,0] * .5 + fmat[:,1] * .5
vocal = fmat[:,1]
master_pow, master_phase = pcm2stft(master)
vocal_pow, vocal_phase = pcm2stft(vocal)
model.fit(master_pow[np.newaxis,:,:stft_len], vocal_pow[np.newaxis,:,:stft_len],
        epochs=100000, batch_size=1, callbacks=[lr, stoppu, btfu])