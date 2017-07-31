import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.signal as signal
import numpy as np
from sklearn.datasets import *

from Layer import *


class Conv2D_layer(Layer):
    def __init__(self, obj, kshape=(1,1,3,3), mode="full", reshape=None, name=None):
        super().__init__(obj, name=name)
        self.obj     = obj.copy()
        self.kshape  = kshape
        self.mode    = mode
        self.reshape = reshape
        self.theta   = theano.shared(np.random.rand(*kshape).astype(dtype=theano.config.floatX), borrow=True)
        self.b       = theano.shared(np.random.rand(1).astype(dtype=theano.config.floatX)[0], borrow=True)
        self.obj.params += [self.theta, self.b]

    def out(self):
        obj = self.obj
        n_in = self.n_in
        
        if self.reshape is not None:
            obj.out = obj.out.reshape(self.reshape)
            n_in = self.reshape

        if obj.out.ndim == 2:
            obj.out = obj.out[None, :, :]
            n_in = (1, n_in[1], n_in[2])
        elif obj.out.ndim == 1:
            obj.out = obj.out[None, None, :]
            n_in = [1, 1] + list(n_in)
            
    
        if self.mode == "full":
            n_out = (self.kshape[0], n_in[-2] + (self.kshape[-2] - 1), n_in[-1] + (self.kshape[-1] - 1))
            obj.out = nnet.conv2d(obj.out, self.theta, border_mode=self.mode) + self.b
        elif self.mode == "valid":
            n_out = (self.kshape[0], n_in[-2] - (self.kshape[-2] - 1), n_in[-1] - (self.kshape[-1] - 1))
            obj.out = nnet.conv2d(obj.out, self.theta, border_mode=self.mode) + self.b
        else:
            n_out   = (self.kshape[0], n_in[-2], n_in[-1])
            h_v, w_v = n_out[-2], n_out[-1]
            h_k, w_k = self.kshape[-2], self.kshape[-1]
            if h_v < h_k:
                h_v, h_k = h_k, h_v
            if w_v < w_k:
                w_v, w_k = w_k, w_v
            h_add = (h_k - 1)
            w_add = (w_k - 1)
            h_left  = h_add // 2
            h_right = h_add - h_left
            w_left  = w_add // 2
            w_right = w_add - w_left
            if h_right == 0: h_right = -1
            if h_left  == 0: h_left = -1
            
            obj.out = nnet.conv2d(obj.out, self.theta, border_mode="full")[:,:, h_left:-h_right, w_left:-w_right] + self.b

        self.n_out = n_out
        
        return obj

    def update(self):
        self.out()
        self.obj.update_node(self.n_out)
        return self.obj


#class Conv2D_layer(Layer):
#    def __init__(self, obj, kshape=(1,1,3,3), mode="full", reshape=None, name=None):
#        super().__init__(obj, name=name)
#        self.obj     = obj.copy()
#        self.kshape  = kshape
#        self.mode    = mode
#        self.reshape = reshape
#        self.theta   = theano.shared(np.random.rand(*kshape).astype(dtype=theano.config.floatX), borrow=True)
#        self.b       = theano.shared(np.random.rand(1).astype(dtype=theano.config.floatX)[0], borrow=True)
#        self.obj.params += [self.theta, self.b]
#
#    def out(self):
#        obj = self.obj
#        n_in = self.n_in
#        
#        if self.reshape is not None:
#            obj.out = obj.out.reshape(self.reshape)
#            n_in = self.reshape
#
#        if obj.out.ndim == 2:
#            obj.out = obj.out[None, :, :]
#            n_in = (1, n_in[1], n_in[2])
#        elif obj.out.ndim == 1:
#            obj.out = obj.out[None, None, :]
#            n_in = [1, 1] + list(n_in)
#    
#        if self.mode == "full":
#            n_out = (self.kshape[0], n_in[-2] + (self.kshape[-2] - 1), n_in[-1] + (self.kshape[-1] - 1))
#        elif self.mode == "valid":
#            n_out = (self.kshape[0], n_in[-2] - (self.kshape[-2] - 1), n_in[-1] - (self.kshape[-1] - 1))
#
#        obj.out = nnet.conv2d(obj.out, self.theta, border_mode=self.mode) + self.b
#
#        self.n_out = n_out
#        
#        return obj
#
#    def update(self):
#        self.out()
#        self.obj.update_node(self.n_out)
#        return self.obj
#