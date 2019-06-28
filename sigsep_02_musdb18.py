#!/usr/bin/env python3
import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# %% imports
import keras
from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
import os, sys
import random
import numpy as np
import math
from stempeg import read_stems
from scipy.io.wavfile import write as wavwrite
from scipy.signal import stft, istft
from glob import glob
from tqdm import tqdm

# %% Constants
blocksize = 1024
overlap = 768  # 3/4
sample_rate = 44100
channel = 2

# %% preprocess, getting the full size
directory = os.path.expanduser("~/Documents/sigsep_data/dsd100/")
filenames = glob(directory+"*.npz")

# %% building the model
from random import shuffle
model = load_model('sigsep_01_mirex-mae-450_epoch-0.22_loss.hdf5')

class DataGenerator_MUSDB18(Sequence):
    def __init__(self, batch_size=32):
        self.filenames = filenames
        self.batch_size = batch_size
    def __len__(self):
        ''' Number of batches per epoch '''
        return len(self.filenames)//self.batch_size
    def __getitem__(self, idx):
        ret_x = np.empty((self.batch_size,513,513,channel))
        ret_y = np.empty((self.batch_size,513,513,channel))
        curr_idx = idx * self.batch_size
        for i in range(self.batch_size):
            data = np.load(self.filenames[curr_idx])
            ret_x[i,:,:,:] = data['train']
            ret_y[i,:,:,:] = data['test']
            curr_idx += 1
        return ret_x, ret_y
    def on_epoch_end(self):
        shuffle(self.filenames)

def get_lrate(epoch, lr):
    return min(.005, .25/(epoch**1.25+1))

loss_type = 'mae'
lr = LearningRateScheduler(get_lrate, verbose=0)
# model.compile(optimizer=RMSprop(lr=.001), loss=loss_type)
# model.compile(optimizer=Adam(lr=.01), loss=loss_type)

# stoppu = EarlyStopping(min_delta=1e-8, patience=5, verbose=1, mode='min')
btfu = ModelCheckpoint('sigsep_03-mae-{epoch:02d}ep-loss_{loss:.5f}.hdf5',
        monitor='loss', save_best_only=False, period=1)

# %% run.
model.fit_generator(generator=DataGenerator_MUSDB18(batch_size=8), workers=4, epochs=2,
        use_multiprocessing=True, callbacks=[btfu, lr])
