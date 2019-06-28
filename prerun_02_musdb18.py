from scipy.signal import stft, istft
import numpy as np
from glob import glob
import math
import stempeg
from os.path import isdir, expanduser
from os import makedirs, removedirs
from tqdm import tqdm

# note: max = 11936

directory = "/home/tran.ngo.quang.ngoc/Downloads/MUS-STEMS-SAMPLE/"
filenames = glob(directory+"train/*.mp4") + glob(directory+"test/*.mp4")

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

expdir = expanduser("~/Documents/sigsep_data/musdb18/")
if not isdir(expdir):
    makedirs(expdir)
count = 0
with tqdm(total=11936) as pbar:
    for fname in filenames:
        stems, _ = stempeg.read_stems(fname, stem_id=[0,4])
        stems = stems.astype(np.float32) * 1024

        master_pow = pcm2stft(stems[0,:,:])
        vocal_pow = pcm2stft(stems[1,:,:])
        curr_pos = 0

        if master_pow.shape[1] < 513:
            pad_size = math.ceil(master_pow.shape[1] / 513) * 513 - master_pow.shape[1]
            master_pow = np.pad(master_pow,
                    ((0,0),(pad_size//2, pad_size - pad_size//2), (0,0)), 'reflect')
            vocal_pow = np.pad(vocal_pow,
                    ((0,0),(pad_size//2, pad_size - pad_size//2), (0,0)), 'reflect')

        while curr_pos + 513 <= master_pow.shape[1]:
            np.savez(expdir+"{:05d}.npz".format(count),
                    train=master_pow[:,curr_pos:curr_pos+513,:],
                    test=vocal_pow[:,curr_pos:curr_pos+513,:])
            curr_pos += 513
            count += 1
            pbar.update(1)
        if curr_pos < master_pow.shape[1]:
            np.savez(expdir+"{:05d}.npz".format(count),
                    train=master_pow[:,-513:,:],
                    test=vocal_pow[:,-513:,:])
            count += 1
            pbar.update(1)