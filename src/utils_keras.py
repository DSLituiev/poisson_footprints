# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial
import numpy as np
from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.utils.np_utils import conv_output_length, conv_input_length

# imports for backwards namespace compatibility
from keras.layers.pooling import AveragePooling1D, AveragePooling2D, AveragePooling3D
from keras.layers.pooling import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.engine import Layer, InputSpec
from keras.regularizers import Regularizer
from keras.constraints import Constraint
from keras.layers import Convolution2D
from keras.layers.advanced_activations import ELU

def softmax2d(x, axis=-1):
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s

class SoftMaxConstr(Constraint):
    '''Constrain the weights to be non-negative.
    '''
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        p = softmax2d(K.log(K.epsilon() + p), axis=self.axis)
        return p

        return {'name': self.__class__.__name__,
                'axis': self.axis}


def pwm_initializer(shape, scale=1.1, name=None, data = None,
                    dim_ordering = K.image_dim_ordering()):
    if data is None:
        logpwm = np.random.rand(*shape)
    else:
        print("shape", shape[:-1])
        print("shape", data.shape)
        assert (data.shape == shape[:-1])
        logpwm = data[:,:,:,None]
    logpwm = K.variable(logpwm, name=name)
    #pwm = softmax2d(logpwm, axis=1)
    return logpwm


class WeightMaxEntropyRegularizer(Regularizer):
    def __init__(self, coef=0.):
        self.coef = K.cast_to_floatx(coef)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            )
        regularized_loss = loss
        regularized_loss += - self.coef * K.sum( self.p * K.log(self.p + K.epsilon()))
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'coef': float(self.coef)}


def pwm_conv(input_img, depth=128, width=3, weights = None, trainable=None,
        constraint = False,
        **kwargs):
    #kwargs = {}
    if weights is not None:
        kwargs["weights"] = [ weights ]
        kwargs["bias"] = False
        kwargs["trainable"] = False
    else:
        kwargs["init"] =  pwm_initializer
    if trainable is not None:
        kwargs["trainable"] = trainable
    kmermap = Convolution2D(depth, width, 1, activation=None,
                      border_mode='same',
                      W_constraint = SoftMaxConstr(axis=1) if constraint else None,
#                       W_regularizer = 
                    **kwargs,
                    )(input_img)
    kmermap = ELU()(kmermap)
    return kmermap

