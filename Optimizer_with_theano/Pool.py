import theano
import theano.tensor.signal as signal
import numpy as np
from sklearn.datasets import *
from .Layer import *

class Pool_layer(Layer):
    def __init__(self, obj, ds=(2,2), name=None):
        super().__init__(obj, name=name)
        self.ds  = ds

    def out(self):
        obj  = self.obj
        n_in = list(self.n_in)
        
        obj.out = signal.pool.pool_2d(obj.out, ds=self.ds, ignore_border=True)
        

        n_in[-2] = n_in[-2] // self.ds[0] #+ (1 if (n_in[-2] % ds[0]) else 0)
        n_in[-1] = n_in[-1] // self.ds[1] #+ (1 if (n_in[-1] % ds[1]) else 0)

        self.n_out = tuple(n_in)
        
    def update(self):
        self.out()
        self.obj.update_node(self.n_out)

        return self.obj

