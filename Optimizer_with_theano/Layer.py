import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.signal as signal
import numpy as np
from sklearn.datasets import *

class Layer():
    def __init__(self, obj, name=None):
        self.obj       = obj
        self.obj.layer_num += 1
        self.n_out = None
        self.obj   = obj.copy()
        
        n_in = obj.layerlst[-1].n_out
        if type(n_in) is not tuple:
            self.n_in  = n_in  = (n_in,) # make tuple
        else:
            self.n_in  = n_in 
        self.name  = name

    def out(self):
        pass

    def update(self):
        pass
    
    def gen_name(self):
        if self.name is None:
            self.name = "Layer_{}".format(self.obj.layer_num)