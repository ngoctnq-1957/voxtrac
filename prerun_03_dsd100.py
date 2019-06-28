from scipy.signal import stft, istft
import numpy as np
from glob import glob
import math
from os.path import isdir, expanduser
from os import makedirs, removedirs
from tqdm import tqdm
from scipy.io.wavfile import read as wavread_orig

rootdir = "/home/tran.ngo.quang.ngoc/Downloads/DSD100/"
folders = glob(rootdir+"Mixtures/Dev/*/") + glob(rootdir+"Mixtures/Test/*/")
folders = [x.replace("/Mixtures/", "/{}/") for x in folders]

# conversions between PCM and STFT
blocksize = 1024
overlap = 768  # 3/4
sample_rate = 44100
stft_params = {'window':'hann', 'nperseg':blocksize,
        'noverlap':overlap, 'nfft':blocksize}

def pcm2stft(pcm: 'wavread') -> {np.ndarray: 'powers', np.ndarray: 'phases'}:
    '''
    takes in a wavfile in matrix form,
    returns power and phase matrices.
    '''
    def stft_help(data):
        _, _, zxx = stft(data, return_onesided=True,
                         padded=False, axis=-1, **stft_params)
        return zxx

    # manual pad the pcm
    if len(pcm.shape) == 1:
        pcm = pcm[:,np.newaxis]
    pad_size = math.ceil(pcm.shape[0]/256)*256 - pcm.shape[0]
    pcm = np.pad(pcm, ((pad_size//2, pad_size-pad_size//2), (0,0)), 'reflect')

    # start processing
    powers = []
    for i in range(pcm.shape[1]):
        stft_temp = stft_help(pcm[:,i])
        # using magnitudes and angles
        powers.append(np.abs(stft_temp))
    return np.dstack(powers)

def wavread(fname):
    rate, data = wavread_orig(fname)
    if data.dtype != np.float32:
        data = data / (np.iinfo(data.dtype).max + 1)
    # don't go vanishing on me
    return rate, data * 2**10

expdir = expanduser("~/Documents/sigsep_data/dsd100/")
if not isdir(expdir):
    makedirs(expdir)
count = 0
for fname in tqdm(folders):
    _, master = wavread(fname.format("Mixtures")+"mixture.wav")
    _, vocal = wavread(fname.format("Sources")+"vocals.wav")
    master_pow = pcm2stft(master)
    vocal_pow = pcm2stft(vocal)
    curr_pos = 0

    pad_size = math.ceil(master_pow.shape[1] / 513) * 513 - master_pow.shape[1]
    master_pow = np.pad(master_pow,
            ((0,0),(pad_size//2, pad_size - pad_size//2), (0,0)), 'reflect')
    vocal_pow = np.pad(vocal_pow,
            ((0,0),(pad_size//2, pad_size - pad_size//2), (0,0)), 'reflect')

    while curr_pos < master_pow.shape[1]:
        np.savez(expdir+"{:05d}.npz".format(count),
                train=master_pow[:,curr_pos:curr_pos+513,:],
                test=vocal_pow[:,curr_pos:curr_pos+513,:])
        curr_pos += 513
        count += 1