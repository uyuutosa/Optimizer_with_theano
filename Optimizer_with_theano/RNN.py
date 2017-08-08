import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.signal as signal
import numpy as np
from sklearn.datasets import *

from .Initializer import *
from .Layer import *
from .Activation import *


class RNN_layer(Layer):
    def __init__(self, 
                 obj, 
                 n_out, 
                 axis=-1,  
                 init_kinds="xavier", 
                 random_kinds="normal", 
                 random_params=(0, 1),
                 is_out=True, 
                 name=None, 
                 activation=None):
        super().__init__(obj, name=name)
        self.obj     = obj.copy()
        self.n_in = n_in = np.array(obj.layerlst[-1].n_out).prod()
        #list(n_in.pop(axis)
        #if not len(n_in):
        #    n_in = (1,)
            
        self.axis = axis
        self.is_out = is_out
        self.activation = activation
        tidx = list(range(np.array(obj.layerlst[-1].n_out).size))
        tidx.pop(axis)
        self.tidx = tidx = np.concatenate([ np.array([0]), np.array(tidx + [self.axis]) + 1])
        self.shape = np.array(obj.layerlst[-1].n_out)
        self.tshape = self.shape[tidx[1:]-1]
        n_rand = self.tshape[:-1].prod()
        if n_rand == 0:
            n_rand = 1
            
        self.theta   = theano.shared(Initializer(name=init_kinds,
                                                 n_in=n_in,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(n_in, n_out),
                                                 ).out.astype(dtype=theano.config.floatX), borrow=True)
        
        self.theta2   = theano.shared(Initializer(name=init_kinds,
                                                 n_in=n_in,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(n_out, n_out),
                                                 ).out.astype(dtype=theano.config.floatX), borrow=True)
        self.b       = theano.shared(Initializer(name=init_kinds,
                                                 n_in=n_in,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(n_out),
                                                 ).out.astype(dtype=theano.config.floatX), borrow=True)
        self.obj.params += [self.theta, self.b]

    def out(self):
        obj = self.obj
        tout = obj.out.transpose(*self.tidx)
       
        arr = tout[..., 0].dot(self.theta)
        if len(self.n_in) == 1:
            arr = arr[...,None]
            
            
        #arr = tout[..., 0:1].dot(self.theta)
        print(arr.shape.eval({obj.x:obj.x_train_arr}))
        lst = [arr]
        for i in range(1, self.n_in):
            arr += Activation(self.activation)(tout[..., i].dot(self.theta)) + self.b + self.theta2.dot(arr)
            #arr += Activation(self.activation)(tout[..., i:i+1].dot(self.theta)) + self.b + self.theta2.dot(arr)
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


