from librosa import load as wavload
from librosa.feature import mfcc

y, sr = wavload('load.wav', sr=None, mono=False)
print(y.shape)
dat = mfcc(y[0,:], sr)
print(dat.shape)
