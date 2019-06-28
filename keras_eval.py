import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

from keras.models import load_model
from scipy.io.wavfile import read as wavread_orig
from scipy.io.wavfile import write as wavwrite
from scipy.signal import stft, istft
import math
import sys
import os
import numpy as np

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
    if len(pcm.shape) == 1:
        pcm = pcm[:,np.newaxis]
    pad_size = math.ceil(pcm.shape[0]/256)*256 - pcm.shape[0]
    pcm = np.pad(pcm, ((pad_size//2, pad_size-pad_size//2), (0,0)), 'reflect')
    
    # start processing
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
    if data.dtype != np.float32 and data.dtype != np.float64:
        data = data / (np.iinfo(data.dtype).max + 1)
    # don't go vanishing on me
    return rate, data * 2**10

if len(sys.argv) < 2:
    raise SystemExit("No input file! Exitting...")

model = load_model('ckpt_02-musdb18-mae-78_epoch-0.15009_loss.hdf5')
# model = load_model('sigsep-mse-07ep-loss_1.18468.hdf5')

data_dir = sys.argv[1]
sample_rate, fmat = wavread(data_dir)

# mixed = fmat[:,0] * .5 + fmat[:,1] * .5
# mixed = fmat
# mixed = mixed[:,np.newaxis]
# fmat = np.hstack([mixed, mixed])
# print(fmat.shape)

master_pow, master_phase = pcm2stft(fmat)
predicted_pow = np.empty_like(master_pow)

length = master_pow.shape[1]
block_size = 513
curr = 0

print('wait -- ', end='')
while curr < length:
    consider = master_pow[:,curr:curr+block_size,:]
    pad = 0
    if consider.shape[1] < block_size:
        pad = block_size - consider.shape[1]
        consider = np.pad(consider, ((0,0), (0,pad), (0,0)), 'constant')
    print('.', end='', flush=True)
    consider = model.predict(consider[np.newaxis,:,:,:])
    predicted_pow[:,curr:curr+block_size-pad,:] = \
            consider[0,:,:-pad,:] if pad > 0 else consider[0,:,:,:]
    curr += block_size
print()

'''
print('right channel ', end='')
curr = 0
while curr < length:
    consider = master_pow_right[:,curr:curr+block_size,:]
    pad = 0
    if consider.shape[1] < block_size:
        pad = block_size - consider.shape[1]
        consider = np.pad(consider, ((0,0), (0,pad), (0,0)), 'constant')
    print('.', end='', flush=True)
    consider = model.predict(consider[np.newaxis,:,:,:])
    predicted_pow_right[:,curr:curr+block_size-pad,:] = \
            consider[0,:,:-pad,:] if pad > 0 else consider[0,:,:,:]
    curr += block_size
print()

predicted_left = stft2pcm(predicted_pow_left, master_phase_left)
predicted_right = stft2pcm(predicted_pow_right, master_phase_right)
predicted = np.hstack([predicted_left, predicted_right])
'''

predicted = stft2pcm(predicted_pow, master_phase)

output_file = 'output.wav'
if len(sys.argv) > 2:
    output_file = sys.argv[2]
wavwrite(output_file, sample_rate, predicted/np.max(np.abs(predicted)))