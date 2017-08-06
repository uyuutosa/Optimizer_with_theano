import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.signal as signal
import numpy as np
from sklearn.datasets import *

from .Layer import *


class RNN_layer(Layer):
    def __init__(self, obj, axis=-1, is_out=True, name=None):
        super().__init__(obj, name=name)
        self.obj     = obj.copy()
        self.axis = axis
        self.is_out = is_out
        tidx = list(range(np.array(obj.layerlst[-1].n_out).size))
        tidx.pop(axis)
        self.tidx = tidx = np.concatenate([ np.array([0]), np.array(tidx + [self.axis]) + 1])
        self.shape = np.array(obj.layerlst[-1].n_out)
        self.tshape = self.shape[tidx[1:]-1]
        n_rand = self.tshape[:-1].prod()
        if n_rand == 0:
            n_rand = 1
        self.theta   = theano.shared(np.random.rand(n_rand, n_rand).astype(dtype=theano.config.floatX), borrow=True)
        self.b       = theano.shared(np.random.rand(1).astype(dtype=theano.config.floatX)[0], borrow=True)
        self.obj.params += [self.theta, self.b]

    def out(self):
        obj = self.obj
        tout = self.obj.out.transpose(self.tidx)
       
        arr = tout[:, 0:1].dot(self.theta)
        lst = [arr]
        for i in range(1, obj.layerlst[-1].n_out[0]):
            arr += tout[:, i:i+1].dot(self.theta) + self.b
            if self.is_out:
                lst += [arr]
            
        if self.is_out:
            obj.out = theano.tensor.concatenate(lst, axis=-1).transpose(self.tidx)
            self.n_out = tuple(self.shape)
        else:
            obj.out = arr[...,None].transpose(self.tidx)
            self.n_out = tuple(self.shape)
        
        return obj

    def update(self):
        self.out()
        self.obj.update_node(self.n_out)
        return self.obj


