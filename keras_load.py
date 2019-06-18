import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

from keras.models import load_model
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.signal import stft, istft
import math
import numpy as np

blocksize = 1024
overlap = 768  # 3/4
sample_rate = 44100

def pcm2stft(pcm: wavread) -> {np.ndarray: 'powers', np.ndarray: 'phases'}:
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

def stft2pcm(powers: np.ndarray, phases: np.ndarray) -> wavread:
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

model = load_model('vox.hdf5')

data_dir = '/home/tran.ngo.quang.ngoc/Downloads/MIR-1K_for_MIREX/Wavfile/'
sample_rate, fmat = wavread(data_dir+'abjones_1_01.wav')
# sample_rate, fmat = wavread(data_dir+'abjones_2_02.wav')

master = fmat[:,0] * .5 + fmat[:,1] * .5
vocal = fmat[:,1]
master_pow, master_phase = pcm2stft(master)

predicted = model.predict(master_pow[np.newaxis,:,:513,:])
audio_out = stft2pcm(predicted[0,:,:,:], master_phase[:,:predicted.shape[2],:])
print(np.max(audio_out))
wavwrite('eval/output.wav', sample_rate, audio_out/np.max(audio_out))
wavwrite('eval/mixed.wav', sample_rate, master/np.max(master))
wavwrite('eval/vocal.wav', sample_rate, vocal/np.max(vocal))