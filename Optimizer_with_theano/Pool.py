import theano
import theano.tensor.signal as signal
import numpy as np
from sklearn.datasets import *
from .Layer import *

class Pool_layer(Layer):
    def __init__(self, 
                 obj, 
                 ds=(2,2), 
                 activation="linear",
                 name=None):
        
        super().__init__(obj, 
                         activation=activation, 
                         name=name)
        self.ds  = ds

    def out(self):
        obj  = self.obj
        n_in = list(self.n_in)
        
        obj.out = signal.pool.pool_2d(obj.out, ds=self.ds, ignore_border=True)

        n_in[-2] = n_in[-2] // self.ds[0] #+ (1 if (n_in[-2] % ds[0]) else 0)
        n_in[-1] = n_in[-1] // self.ds[1] #+ (1 if (n_in[-1] % ds[1]) else 0)

        self.n_out = tuple(n_in)

    def gen_name(self):
        if self.name is None:
            self.name = "Pool_{}".format(self.obj.layer_info.layer_num)
            
class Unpool_layer(Layer):
    def __init__(self, 
                 obj, 
                 ds=(2,2), 
                 activation="linear",
                 name=None):
        
        super().__init__(obj, 
                         activation=activation, 
                         name=name)
        self.ds  = ds

    def out(self):
        obj  = self.obj
        n_in = list(self.n_in)
        shape = obj.out.shape
        #shape[-1] = shape[-1] * self.ds[-1]
        #shape[-2] = shape[-2] * self.ds[-2]
        
        obj.out = obj.out[...,None,None] * T.ones((2, 2))
        obj.out = obj.out.reshape((shape[0], shape[1], shape[2]*self.ds[-2], shape[3]*self.ds[-1]))

        n_in[-1] = n_in[-1] * self.ds[-1]
        n_in[-2] = n_in[-2] * self.ds[-2]

        self.n_out = tuple(n_in)

    def gen_name(self):
        if self.name is None:
            self.name = "Unpool_{}".format(self.obj.layer_info.layer_num)