# %% Imports
import tensorflow as tf
import numpy as np
import math
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.signal import stft, istft

# %% Constants
blocksize = 1024
overlap = 768  # 3/4
sample_rate = 44100

# %% make STFT from PCM
def pcm2stft(pcm: wavread) -> tf.Tensor:
    def stft_help(data):
        _, _, zxx = stft(data, window='hann', nperseg=blocksize,
                         noverlap=overlap, nfft=blocksize, return_onesided=True,
                         padded=False, axis=-1)
        # print(f.shape, t.shape, zxx.shape)
        return zxx
    # manual pad the pcm
    pad_size = math.ceil(pcm.shape[0]/256)*256 - pcm.shape[0]
    pcm = np.pad(pcm, (pad_size, 0), 'reflect')
    # start processing
    if len(pcm.shape) == 1:
        pcm = pcm[:,np.newaxis]
    acc = []
    for i in range(pcm.shape[1]):
        stft_temp = stft_help(pcm[:,i])
        # NOTE: may use real/imag or mag/angle
        acc.append(np.real(stft_temp))
        acc.append(np.imag(stft_temp))
    return np.dstack(acc)

# %% Setup
batch_size = 10
channels = 1
stft_size = 1000

# %% model stealing whores
tf.compat.v1.reset_default_graph()
xavier = tf.initializers.glorot_normal()
dimensions = [None, 513, stft_size, 2*channels]
stft_input = tf.compat.v1.placeholder(tf.float32, dimensions)
stft_output = tf.compat.v1.placeholder(tf.float32, dimensions)

conv1_mat1 = tf.Variable(xavier([5,5,2*channels,4*channels]), dtype=tf.float32)
x = tf.nn.conv2d(stft_input, filter=conv1_mat1, strides=(1,1,1,1), padding='VALID')
# removed batch normalization
x = tf.nn.leaky_relu(x, 0.25)
conv1_mat2 = tf.Variable(xavier([5,5,4*channels,4*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv1_mat2, strides=(1,1,1,1), padding='VALID')
skip1 = tf.nn.leaky_relu(x, 0.25)
x = tf.nn.pool(skip1, [2,2], 'MAX', 'VALID', strides=[1,2])
# college
x = tf.nn.dropout(x, rate=.1)

conv2_mat1 = tf.Variable(xavier([3,3,4*channels,8*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv2_mat1, strides=(1,1,1,1), padding='VALID')
# removed batch normalization
x = tf.nn.leaky_relu(x, 0.25)
conv2_mat2 = tf.Variable(xavier([3,3,8*channels,8*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv2_mat2, strides=(1,1,1,1), padding='VALID')
skip2 = tf.nn.leaky_relu(x, 0.25)
x = tf.nn.pool(skip2, [2,2], 'MAX', 'VALID', strides=[1,2])
x = tf.nn.dropout(x, rate=.1)

conv3_mat1 = tf.Variable(xavier([3,3,8*channels,16*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv3_mat1, strides=(1,1,1,1), padding='VALID')
# removed batch normalization
x = tf.nn.leaky_relu(x, 0.25)

conv3_mat2 = tf.Variable(xavier([3,3,16*channels,16*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv3_mat2, strides=(1,1,1,1), padding='VALID')
x = tf.nn.leaky_relu(x, 0.25)
x = tf.nn.dropout(x, rate=.1)

dim1 = int(x.shape[1])+2
dim2 = int(x.shape[2])*2+1
upconv2_mat = tf.Variable(xavier([3,3,8*channels,16*channels]), dtype=tf.float32)
x = tf.nn.conv2d_transpose(x, filter=upconv2_mat,
        output_shape=[batch_size,dim1,dim2,8*channels], strides=(1,1,2,1), padding='VALID')

start1 = (skip2.shape[1]-dim1)//2
start2 = (skip2.shape[2]-dim2)//2

# skip connection
x = tf.concat((skip2[:,start1:start1+dim1,start2:start2+dim2,:], x), -1)

conv2_mat3 = tf.Variable(xavier([3,3,16*channels,8*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv2_mat3, strides=(1,1,1,1), padding='VALID')
x = tf.nn.leaky_relu(x, 0.25)

conv2_mat4 = tf.Variable(xavier([3,3,8*channels,8*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv2_mat4, strides=(1,1,1,1), padding='VALID')
x = tf.nn.leaky_relu(x, 0.25)
x = tf.nn.dropout(x, rate=.1)

dim1 = int(x.shape[1])+2
dim2 = int(x.shape[2])*2+1
upconv1_mat = tf.Variable(xavier([3,3,4*channels,8*channels]), dtype=tf.float32)
x = tf.nn.conv2d_transpose(x, filter=upconv1_mat,
        output_shape=[batch_size,dim1,dim2,4*channels], strides=(1,1,2,1), padding='VALID')

start1 = (skip1.shape[1]-dim1)//2
start2 = (skip1.shape[2]-dim2)//2

# skip connection
x = tf.concat((skip1[:,start1:start1+dim1,start2:start2+dim2,:], x), -1)

conv1_mat3 = tf.Variable(xavier([3,3,8*channels,4*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv1_mat3, strides=(1,1,1,1), padding='VALID')
x = tf.nn.leaky_relu(x, 0.25)

conv1_mat4 = tf.Variable(xavier([3,3,4*channels,4*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=conv1_mat4, strides=(1,1,1,1), padding='VALID')
x = tf.nn.leaky_relu(x, 0.25)
x = tf.nn.dropout(x, rate=.1)

last_conv_mat = tf.Variable(xavier([1,1,4*channels,2*channels]), dtype=tf.float32)
x = tf.nn.conv2d(x, filter=last_conv_mat, strides=(1,1,1,1), padding='VALID')

res_shape = x.shape
start_1 = (513 - res_shape[1]) // 2
start_2 = (stft_size - res_shape[2]) // 2

x = tf.pad(x, [[0,0], [start_1, 513 - res_shape[1] - start_1],
                [start_2, stft_size - res_shape[2] - start_2], [0,0]], mode='CONSTANT')

learning_rate = tf.compat.v1.placeholder(tf.float32, [])
loss = tf.compat.v1.losses.mean_squared_error(x, stft_output)
lego = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(loss)

# %% If this is wrong I would be very devastated
import glob, random, sys, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def animate(i):
    ax1.clear()
    ax1.plot(t, lplot)
t = []
lplot = []
plt.show()

losses = []
filenames = glob.glob("/home/tran.ngo.quang.ngoc/Downloads/MIR-1K_for_MIREX/Wavfile/*.wav")
saver = tf.compat.v1.train.Saver()

stft_i = np.empty((batch_size, 513, stft_size, 2*channels))
stft_o = np.empty((batch_size, 513, stft_size, 2*channels))

rate = lambda: min(.01, .5/(len(losses)+1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while os.path.isfile('go'):
        count = 0
        while count < batch_size:
            fname = random.choice(filenames)
            sample_rate, waves = wavread(fname)
            acapella = pcm2stft(waves[:,1])
            mastered = pcm2stft(waves[:,0]*.5 + waves[:,1]*.5)
            
            length = acapella.shape[1]
            if length < stft_size:
                stft_i[count,:,:,:] = np.pad(mastered,
                        ((0,0), (0, stft_size-length), (0,0)), 'constant')
                stft_o[count,:,:,:] = np.pad(mastered,
                        ((0,0), (0, stft_size-length), (0,0)), 'constant')
                count += 1
            else:
                while length > 0:
                    if count == batch_size:
                        break
                    mastered1 = mastered[:,:,length:length+stft_size]
                    acapella1 = acapella[:,:,length:length+stft_size]
                    if length < stft_size:
                        stft_i[count,:,:] = np.pad(mastered1,
                                ((0,0), (0,0), (0, stft_size-length)), 'constant')
                        stft_o[count,:,:] = np.pad(mastered1,
                                ((0,0), (0,0), (0, stft_size-length)), 'constant')
                        length = 0
                    else: length -= stft_size
                    count += 1

        # max_val = np.max(np.abs(stft_i)), np.max(np.abs(stft_o))
        # max_val = max(max_val) / 10

        _, lossval = sess.run([lego, loss], feed_dict= \
                {stft_input: stft_i,
                stft_output: stft_o,
                learning_rate: rate()})
        losses.append(lossval)
        saver.save(sess, "/tmp/model.ckpt")

        # plotting
        real_len = min(50, len(losses))
        t = list(map(lambda x: x+len(losses)-real_len, range(real_len)))
        lplot = losses[len(losses) - real_len:]
        plt.clf()
        plt.plot(t, lplot, 'b')
        plt.title("Current loss: {}".format(lossval))
        plt.pause(0.1)