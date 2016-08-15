#!/usr/bin/env python3
"purge theano cache if 'no test value' errors appear"
#! theano-cache purge
# !pip3 install keras
# import os
# import sys
import numpy as np
# import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.objectives import mean_squared_error
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import merge
from keras.models import Model
from keras.constraints import MaxNorm, UnitNorm
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import ActivityRegularizer

from utils_keras import pwm_conv
# from keras.callbacks import TensorBoard

def log_poisson(y_true, log_y_pred):
    return K.mean(K.exp(log_y_pred) - y_true * log_y_pred, axis=-1)

def poi_gau_mix(y_true, log_y_pred):
    return log_poisson(y_true, log_y_pred) + 0.01*mean_squared_error(y_true, K.exp(log_y_pred))

def exp_init(shape, name=None):
    value = np.log(np.random.rand(*shape)) *(1-2*(np.random.rand(*shape)>0.5))
    return K.variable(value, name=name)


DIM_ORDERING = 'th' # 'th' (channels, width, height) or 'tf' (width, height, channels)
if DIM_ORDERING == 'th':
    img_input = Input(shape=(3, 299, 299))
    CONCAT_AXIS = 1
elif DIM_ORDERING == 'tf':
    img_input = Input(shape=(299, 299, 3))
    CONCAT_AXIS = 3
else:
    raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))


def create_network(xdepth = 4, xlen = 2000):
    input_img = Input(shape=(xdepth, xlen, 1))

    kmermap11 = pwm_conv(input_img, depth=128, width=11)
    kmermap5 = pwm_conv(input_img, depth=64, width=5)

    kmermaps = merge([kmermap11, kmermap5], mode='concat', concat_axis=CONCAT_AXIS)

    bypass = Convolution2D(32, 1, 1, activation=None, border_mode='same')(kmermaps)
    bypass = BatchNormalization(epsilon=1e-06, mode=0, axis=1,
                           momentum=0.99, weights=None, beta_init='zero', gamma_init='one')(bypass)
    ###########################################################
    # x = MaxPooling2D((2, 1), border_mode='same')(x)
    x = Convolution2D(64, 5, 1, activation=None,
                      activity_regularizer = ActivityRegularizer(l1=5e-7, l2=1e-6),
                      border_mode='same')(kmermaps)
    x = BatchNormalization(epsilon=1e-06, mode=0, axis=1,
                           momentum=0.99, weights=None, beta_init='zero', gamma_init='one')(x)
    x = ELU()(x)
    # x = MaxPooling2D((2, 2), border_mode='same')(x)
    # x = UpSampling2D((2, 1))(x)
    x = Convolution2D(32, 3, 1, activation=None, border_mode='same')(x)
    x = BatchNormalization(epsilon=1e-06, mode=0, axis=1,
                           momentum=0.99, weights=None, beta_init='zero', gamma_init='one')(x)
    x = ELU()(x)
    ###########################################################
    x = merge([x, bypass], mode='concat', concat_axis=CONCAT_AXIS)

    # x = MaxPooling2D((2, 1), border_mode='same')(x)
    x = Convolution2D(64, 5, 1, activation=None,
                      activity_regularizer = ActivityRegularizer(l1=5e-7, l2=1e-6),
                      border_mode='same')(x)
    x = BatchNormalization(epsilon=1e-06, mode=0, axis=1,
                           momentum=0.99, weights=None, beta_init='zero', gamma_init='one')(x)
    x = ELU()(x)
    # x = MaxPooling2D((2, 2), border_mode='same')(x)
    # x = UpSampling2D((2, 1))(x)
    x = Convolution2D(8, 3, 1, activation=None, border_mode='same')(x)
    x = BatchNormalization(epsilon=1e-06, mode=0, axis=1,
                           momentum=0.99, weights=None, beta_init='zero', gamma_init='one')(x)
    x = ELU()(x)
    ###########################################################
    decoded = Convolution2D(1, 1, 1, activation=None, border_mode='same')(x)

    mo = Model(input_img, decoded)
    return mo

##################################################################################
##################################################################################
BATCH_SIZE = 2**6
XLEN = 2000
"paths to the data sets"
dbdir = "../data/"
dbpath = dbdir + "batf_disc1_gw.db"
"path to the model"
filepath = "mo1keras_batchnorm_unitnorm_5_11"

mo = create_network(xdepth = 4, xlen=XLEN)
mo.compile(optimizer='adadelta', loss=log_poisson)
try:
    mo.load_weights(filepath)
except OSError:
    print("no weights found")
mo.optimizer.lr.set_value(.5)

from keras.callbacks import LearningRateScheduler
def scheduler(epoch):
    if epoch == 15:
        mo.optimizer.lr.set_value(.1)
    if epoch == 20:
        mo.optimizer.lr.set_value(.05)
    return float(mo.optimizer.lr.get_value())

change_lr = LearningRateScheduler(scheduler)

############################################################
import sqlite3
from match_dna_atac import get_loader, align_shapes, get_seq_random
conn = sqlite3.connect(dbpath, check_same_thread=False)

train_batchloader = get_loader(conn, fraction= -1/8, binary=False)
test_batchloader  = get_loader(conn, fraction=  1/32, binary=False)
def keras_loader_wrap(loaderfun, BATCH_SIZE, xlen):
    while True:
        loader = loaderfun(BATCH_SIZE)
        xx_, yy_ = next(loader)
        for xx, yy in (loader):
            xx = np.vstack([xx_,xx])
            yy = np.vstack([yy_,yy])

            mixer = np.zeros(xx.shape[0], dtype=bool)
            mixer[np.random.permutation(xx.shape[0])[:xx.shape[0]//2]] = True

            xx_ = xx[~mixer]
            yy_ = yy[~mixer]
            yield (xx[mixer].transpose(0,3,2,1)[:,:,:xlen,:], yy[mixer][:,None,:,None][:,:,:xlen,:])
        yield (xx_.transpose(0,3,2,1)[:,:,:xlen,:], yy_[:,None,:,None][:,:,:xlen,:])

def get_samples_per_epoch(loader, BATCH_SIZE):
    ncycles = 0
    for _ in loader:
        ncycles +=1
    return BATCH_SIZE*ncycles

SAMPLES_PER_EPOCH = get_samples_per_epoch(train_batchloader(BATCH_SIZE), BATCH_SIZE)

# for x,y in keras_loader_wrap(train_batchloader, BATCH_SIZE):
#     print(x.shape, y.shape)
#     break

try:
    for _ in range(50):
        mo.fit_generator(keras_loader_wrap(train_batchloader, BATCH_SIZE, xlen=XLEN),
                         nb_epoch=20, samples_per_epoch = SAMPLES_PER_EPOCH,
                         callbacks=[change_lr])
except Exception as ee:
    raise ee
finally:
    mo.save(filepath)

############################################################
############################################################

bs = 128
yhat = mo.predict_generator(keras_loader_wrap(test_batchloader,bs), val_samples=bs)
import numpy as np
import matplotlib.pyplot as plt
T = yhat[0,0].shape[0]
t = np.arange(-T//2, T//2)
fig, ax = plt.subplots(1, figsize= 1.5*np.r_[6,4])
plt.plot(t, np.exp(yhat[:,0].T[0]), c=(1,0,0,0.2))
plt.plot(t, np.exp(yhat[:,0].T[0].mean(1)), "-", c=(0,0,.8,1), lw=2, label="mean predicted")
plt.plot(t, np.mean(y_, axis=0).squeeze(), c=(0,.75,0,1), lw=2, label="mean observed")
plt.legend()
plt.xlim( [-20, 20])


# In[45]:

T = yhat[0,0].shape[0]
t = np.arange(-T//2, T//2)
fig, ax = plt.subplots(1, figsize= 1.5*np.r_[6,4])
plt.plot(t, np.exp(yhat[:,0].T[0]), c=(1,0,0,0.2))
plt.plot(t, np.exp(yhat[:,0].T[0].mean(1)), "-", c=(0,0,.8,1), lw=2, label="mean predicted")
plt.plot(t, np.mean(y_, axis=0).squeeze(), c=(0,.75,0,1), lw=2, label="mean observed")
plt.legend()
plt.xlim( [500, 600])


T = yhat[0,0].shape[0]
t = np.arange(-T//2, T//2)
# t.shape
fig, ax = plt.subplots(1, figsize= 1.5*np.r_[6,4])
plt.plot(t, y_.mean(0).squeeze(), c=(0,.75,0,1), label="mean observed")
plt.plot(t, np.squeeze(y_).T, c=(1,0,0,0.2))
plt.plot(t, np.exp(yhat[:,0].T[0].mean(1)), "-", c=(0,0,.8,1), lw=2, label="mean predicted")
plt.ylim([0,80])
plt.xlim( [-20, 20])


# In[29]:

nn = 8
plt.plot([0,10], [0,10])
plt.scatter(np.exp(yhat[nn,0,:,0]), y_[nn,0,:,0],
            c=(0,.75,0,0.1), edgecolors='none')
plt.ylabel("observed")
plt.xlabel("predicted")


# In[29]:

la = mo.layers[1]
la.get_weights()[0].shape


# In[30]:

la.get_output_shape_at(0)


# In[45]:

w = la.get_weights()[0]
print(w.shape)
# (w[0,:,:,0]**2).sum(0)
w[0,:,:,0]


# In[37]:

y_.mean(0).squeeze().shape

radius=1000
valid = abs(t)<radius
fig, axs = plt.subplots(2, figsize= 1.5*np.r_[6,4])
axs[0].plot(t[valid], y_.mean(0).squeeze()[valid], c=(0,.75,0,1), lw=2, label="mean observed",)
axs[1].plot(t[valid], y_.var(0).squeeze()[valid], "-", c=(.5,0,0,1), lw=2, label="var observed",)
[ax.legend() for ax in axs]
plt.xlim( [-radius, radius])

