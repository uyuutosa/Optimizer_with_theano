import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.signal as signal
import numpy as np
from sklearn.datasets import *
from .Activation import Activation

class Layer():
    def __init__(self, obj, activation="linear", name=None):
        self.obj       = obj
        self.n_out = None
        self.obj   = obj.copy()
        
        n_in = obj.layer_info.get_shape_of_last_node()
        if type(n_in) is not tuple:
            self.n_in  = n_in  = (n_in,) # make tuple
        else:
            self.n_in  = n_in 
        self.name  = name
        self.params = []
        self.gen_name()
        self.act = Activation(activation)
        self.actname = activation
        self.params = {}

    def out(self):
        pass

    def update(self, obj=None):
        if obj is not None:
            self.obj = obj
        self.out()
        self.obj.out = self.act(self.obj.out)
        self.obj.layer_info.set_layer(self)
        return self.obj
    
    def gen_name(self):
        if self.name is None:
            self.name = "Layer_{}".format(self.obj.layer_info.layer_num)
            
    def get_params(self):
        return self.params