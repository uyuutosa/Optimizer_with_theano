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
                 axis=0,  
                 init_kinds="xavier", 
                 random_kinds="normal", 
                 random_params=(0, 1),
                 is_out=False, 
                 is_train=True,
                 activation=None,
                 name=None):
        super().__init__(obj, 
                         activation=activation,
                         name=name)
        self.obj = obj.copy()
        self.n_in = n_in = obj.layerlst[-1].n_out
        self.n_iter = n_in[axis]
        self.n_out = (n_out,)
        if len(n_in) == 1:
            n_in = 1
            
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
                                                 n_in=n_out,
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
        
        #self.obj.params += [self.theta, self.theta2, self.b]
        if is_train:
            self.params = {self.name + "_theta":self.theta, 
                           self.name + "_theta2":self.theta2, 
                           self.name + "_b":self.b} 
        else:
            self.params = {}

    def out(self):
        obj = self.obj
        tout = obj.out.transpose(*self.tidx)
        self.obj.tout = tout
        
       
        if len(self.n_in) == 1:
            def _step(seq, prior):
                y = tout[..., seq:seq+1].dot(self.theta) + self.b + prior.dot(self.theta2)
                return Activation(self.activation)(y)
            i = theano.shared(self.n_iter)
            
            arr = T.zeros_like(tout[..., 0:1]).dot(self.theta)
            self.obj.arr = arr
            result, updates = theano.scan(fn=_step,
                              sequences=T.arange(i),
                              outputs_info=arr,
                              non_sequences=None,
                              n_steps=None)
            self.obj.result = result
        else:
            def _step(seq, prior):
                y = tout[..., seq].dot(self.theta) + self.b + prior.dot(self.theta2)
                return Activation(self.activation)(y)
            i = theano.shared(self.n_iter)
            
            arr = T.zeros_like(tout[..., 0]).dot(self.theta)
            self.obj.arr = arr
            result, updates = theano.scan(fn=_step,
                              sequences=T.arange(i),
                              outputs_info=arr,
                              non_sequences=None,
                              n_steps=None)
            self.obj.result = result
            
        if self.is_out:
            obj.out = result
            #obj.out = theano.tensor.concatenate(lst, axis=-1).transpose(self.tidx)
            #self.n_out = 
            #self.n_out = tuple(self.shape)
        else:
            obj.out = result[-1]
            #obj.out = T.unbroadcast(result[...,-1], 0,1)
            #obj.out = arr.transpose(self.tidx)
            #obj.out = arr[...,None].transpose(self.tidx)
            #self.n_out = tuple(self.shape)
            #self.n_out = tuple(self.shape)

    def gen_name(self):
        if self.name is None:
            self.name = "RNN_{}".format(self.obj.layer_num)

"""
class LSTM_layer(Layer):
    def __init__(self, 
                 obj, 
                 n_out, 
                 axis=0,  
                 init_kinds="xavier", 
                 random_kinds="normal", 
                 random_params=(0, 1),
                 is_out=False, 
                 name=None, 
                 activation=None):
        super().__init__(obj, name=name)
        self.obj = obj.copy()
        self.n_in = n_in = obj.layerlst[-1].n_out
        self.n_iter = n_in[axis]
        self.n_out = (n_out,)
        if len(n_in) == 1:
            n_in = 1
            
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
                                                 n_in=n_out,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(n_out, n_out),
                                                  
        self.theta3   = theano.shared(Initializer(name=init_kinds,
                                                 n_in=n_out,
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
        self.b2 = theano.shared(Initializer(name=init_kinds,
                                                 n_in=n_in,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(n_out),
                                                 ).out.astype(dtype=theano.config.floatX), borrow=True)
        
        self.obj.params += [self.theta, self.theta2, self.b]

    def out(self):
        obj = self.obj
        tout = obj.out.transpose(*self.tidx)
        self.obj.tout = tout
        
       
        if len(self.n_in) == 1:
            def _step(seq, prior):
                y = tout[..., seq:seq+1].dot(self.theta) + self.b + prior.dot(self.theta2)
                return Activation(self.activation)(y)
            i = theano.shared(self.n_iter)
            
            arr = T.zeros_like(tout[..., 0:1]).dot(self.theta)
            self.obj.arr = arr
            result, updates = theano.scan(fn=_step,
                              sequences=T.arange(i),
                              outputs_info=arr,
                              non_sequences=None,
                              n_steps=None)
            self.obj.result = result
        else:
            def _step(seq, prior):
                y = tout[..., seq].dot(self.theta) + self.b + prior.dot(self.theta2)
                return Activation(self.activation)(y)
            i = theano.shared(self.n_iter)
            
            arr = T.zeros_like(tout[..., 0]).dot(self.theta)
            self.obj.arr = arr
            result, updates = theano.scan(fn=_step,
                              sequences=T.arange(i),
                              outputs_info=arr,
                              non_sequences=None,
                              n_steps=None)
            self.obj.result = result
            
        if self.is_out:
            obj.out = result
            #obj.out = theano.tensor.concatenate(lst, axis=-1).transpose(self.tidx)
            #self.n_out = 
            #self.n_out = tuple(self.shape)
        else:
            obj.out = result[-1]
            #obj.out = T.unbroadcast(result[...,-1], 0,1)
            #obj.out = arr.transpose(self.tidx)
            #obj.out = arr[...,None].transpose(self.tidx)
            #self.n_out = tuple(self.shape)
            #self.n_out = tuple(self.shape)
        
        return obj

    def update(self):
        self.out()
        self.obj.update_node(self.n_out)
        return self.obj

"""