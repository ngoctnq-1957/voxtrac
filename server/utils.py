import logging, warnings
from scipy.signal import stft, istft
import math
import sys
import os
import numpy as np
import json
import librosa, requests
import grpc
import request
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

blocksize = 1024
overlap = 768  # 3/4
sample_rate = 44100
block_size = 513
# model = load_model('ckpt_02-musdb18-mae-78_epoch-0.15009_loss.hdf5')

SERVER_URL = 'http://localhost:26895/extracted_model/voxtrac:predict'

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
    data, rate = librosa.core.load(fname, sr=None, mono=False)
    data = data.T
    if data.dtype not in [np.float32, np.float64]:
        data = data / (np.iinfo(data.dtype).max + 1)
    # don't go vanishing on me
    return rate, data * 2**10

def get_vox(fname_in, fname_out, ext='wav'):
    sample_rate, fmat = wavread(fname_in)
    master_pow, master_phase = pcm2stft(fmat)
    predicted_pow = np.empty_like(master_pow)

    length = master_pow.shape[1]
    curr = 0

    while curr < length:
        consider = master_pow[:,curr:curr+block_size,:]
        pad = 0
        if consider.shape[1] < block_size:
            pad = block_size - consider.shape[1]
            consider = np.pad(consider, ((0,0), (0,pad), (0,0)), 'constant')
        consider = consider[np.newaxis,:,:,:]
        # consider = model.predict(consider)

        togo = json.dumps(consider.tolist())

        channel = grpc.insecure_channel('localhost:26895')
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'voxtrac'
        request.model_spec.signature_name = 'serving_default'

        print('Sending request...')
        response = requests.post(SERVER_URL, data=togo)
        print('Waiting for response...')
        response.raise_for_status()
        print('Got it...')
        print(response.json())
        # np.asarray(json.loads(togo))
        
        predicted_pow[:,curr:curr+block_size-pad,:] = \
                consider[0,:,:-pad,:] if pad > 0 else consider[0,:,:,:]
        curr += block_size

    predicted = stft2pcm(predicted_pow, master_phase)
    print('Writing to', fname_out)
    librosa.output.write_wav(fname_out, predicted, sample_rate, norm=True)