import numpy as np
from scipy.io.wavfile import read as wavread_orig
from scipy.io.wavfile import write as wavwrite
from scipy.signal import stft, istft
import math
import sys

# Constants
blocksize = 1024
overlap = 768  # 3/4
sample_rate = 44100

# conversions between PCM and STFT
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
        _, zxx = istft(data, input_onesided=True, **stft_params)
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

if len(sys.argv) < 3:
    raise SystemExit("Usage: {} <input_file> <volume_magnitude>".format(sys.argv[0]))
fname = sys.argv[1]
mag = sys.argv[2]
try:
    rate, data = wavread(fname)
except:
    raise SystemExit("File does not exist!")
try:
    mag = float(mag)
except:
    raise SystemExit("Not a valid magnitude!")

# print('Transforming...')
# real, fake = pcm2stft(data)
# print('Boosting the all freqs...')
# real[:,:,:] = real[:,:,:] / 2
# data = stft2pcm(real, fake)
data *= mag
print('Hard clipping data...')
data = np.maximum(np.minimum(data, 1), -1)
print('Exporting out into \'output.wav\'...')
wavwrite('output.wav', rate, data)
print('Done!')