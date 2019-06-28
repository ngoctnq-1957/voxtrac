#!/usr/bin/env python3
import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# %% imports
import keras
from keras.layers import Input, SeparableConv2D, LeakyReLU, Dropout, \
        UpSampling2D, Concatenate, Cropping2D, MaxPooling2D, ZeroPadding2D, \
        BatchNormalization, DepthwiseConv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model, print_summary, Sequence
import os, sys
import random
import operator
import numpy as np
import math
from scipy.io.wavfile import write as wavwrite
from utils import *

# %% building the model
relu_rate = .2
drop_rate = .1
# for now
channel = 2
# HACK: DO NOT CHANGE THIS IT WILL BREAK THE CODE -- LOOK INTO DATAGENERATOR
stft_len = 513

main_input = Input(shape=(513,stft_len,channel), dtype='float32', name='audio_input')
# convolution 1
x = SeparableConv2D(filters=2*channel, kernel_size=5, padding='SAME')(main_input)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=4*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = BatchNormalization()(x)
skip1 = Dropout(rate=drop_rate)(x)

# Pooling
x = MaxPooling2D(pool_size=4, strides=2)(skip1)

# convolution 2
x = SeparableConv2D(filters=8*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=16*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = BatchNormalization()(x)
skip2 = Dropout(rate=drop_rate)(x)

# Pooling
x = MaxPooling2D(pool_size=4, strides=2)(skip2)

# convolution 3
x = SeparableConv2D(filters=32*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=64*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = BatchNormalization()(x)
skip3 = Dropout(rate=drop_rate)(x)

# Pooling
x = MaxPooling2D(pool_size=4, strides=2)(skip3)

# bottom layer, do not expand
x = DepthwiseConv2D(kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = DepthwiseConv2D(kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = BatchNormalization()(x)

# Deconvolve to match dimensions
x = Conv2DTranspose(filters=64*channel, kernel_size=4, strides=2)(x)
x = Concatenate(axis=-1)([skip3, x])

# and convolve
x = SeparableConv2D(filters=32*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=16*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = BatchNormalization()(x)

x = Conv2DTranspose(filters=16*channel, kernel_size=5, strides=2)(x)
x = Concatenate(axis=-1)([skip2, x])
# and convolve
x = SeparableConv2D(filters=8*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=4*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = BatchNormalization()(x)

x = Conv2DTranspose(filters=4*channel, kernel_size=5, strides=2)(x)
x = Concatenate(axis=-1)([skip1, x])
# and convolve
x = SeparableConv2D(filters=2*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)
x = SeparableConv2D(filters=1*channel, kernel_size=5, padding='SAME')(x)
x = LeakyReLU(alpha=relu_rate)(x)

main_output = x
model = Model(inputs=main_input, outputs=main_output)
plot_model(model, 'model.png')
print_summary(model)

# %% some additional model helpers
from glob import glob
from random import shuffle
directory = "/home/tran.ngo.quang.ngoc/Downloads/MIR-1K_for_MIREX/Wavfile/"

class DataGenerator_MIREX(Sequence):
    def __init__(self, batch_size=32):
        self.filenames = glob(directory+"*.wav")
        self.batch_size = batch_size
    def __len__(self):
        ''' Number of batches per epoch '''
        return len(self.filenames) // self.batch_size
    def __getitem__(self, idx):
        batch_files = self.filenames[idx*self.batch_size:(idx+1)*self.batch_size]
        ret_x = np.empty((self.batch_size,513,513,channel))
        ret_y = np.empty((self.batch_size,513,513,channel))
        for i in range(self.batch_size):
            _, fmat = wavread(batch_files[i])
            master = fmat[:,0] * .5 + fmat[:,1] * .5
            vocal = fmat[:,1]
            master_pow, _ = pcm2stft(master)
            vocal_pow, _ = pcm2stft(vocal)
            # pad if needed
            if master_pow.shape[1] < 513:
                left_pad = (513 - master_pow.shape[1])//2
                right_pad = 513 - master_pow.shape[1] - left_pad
                master_pow = np.pad(master_pow,
                        ((0,0),(left_pad, right_pad), (0,0)), 'reflect')
                vocal_pow = np.pad(vocal_pow,
                        ((0,0),(left_pad, right_pad), (0,0)), 'reflect')
            # cropping the length to fit the STFT max size
            left_start = (master_pow.shape[1] - 513) // 2
            master_pow = master_pow[:,left_start:left_start+513,:]
            vocal_pow = vocal_pow[:,left_start:left_start+513,:]
            # and fill
            ret_x[i,:,:,:1] = master_pow
            ret_y[i,:,:,:1] = vocal_pow
            ret_x[i,:,:,1:] = master_pow
            ret_y[i,:,:,1:] = vocal_pow
        return ret_x, ret_y
    def on_epoch_end(self):
        shuffle(self.filenames)

def get_lrate(epoch, lr):
    ''' learning rate
    drop = 0.5
    epochs_drop = 2.0
    lrate = lr * math.pow(drop, 
        math.floor((1+epoch)/epochs_drop))
    '''
    # try jumping out of local min shall there exists
    # if random.random() < .1:
    #     return lr * 2
    return min(.005, .25/(epoch**1.25+1))

loss_type = 'mae'
lr = LearningRateScheduler(get_lrate, verbose=0)
# model.compile(optimizer=RMSprop(lr=.001), loss=loss_type)
model.compile(optimizer=Adam(lr=.01), loss=loss_type)
# model.compile(optimizer='adam', loss=loss_type)

stoppu = EarlyStopping(min_delta=1e-8, patience=5, verbose=1, mode='min')
btfu = ModelCheckpoint('vox_'+loss_type+'.hdf5', save_best_only=False, period=1)

# %% real shit
model.fit_generator(generator=DataGenerator_MIREX(batch_size=2), workers=4, epochs=1000,
        use_multiprocessing=True, callbacks=[stoppu, btfu, lr])