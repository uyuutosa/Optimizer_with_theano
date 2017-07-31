from Layer import *
#import theano
#import theano.tensor as T
#import theano.tensor.nnet as nnet
#import theano.tensor.signal as signal
#import numpy as np
#from sklearn.datasets import *

class Dropout():
    def __init__(self, obj, rate, name=None):
        super().__init__(obj, name=name)
        self.rate = theano.shared(rate)

    def out(self):
        obj     = self.obj
        obj.out = T.where(srng.uniform(size=obj.train_out.shape) > self.rate, obj.out, 0)
        return obj

    def update(self):
        self.out()
        self.obj.update_node([self.n_out])
        return self.obj
    
    def gen_name(self):
        if self.name is None:
            self.name = "Dense_{}".format(self.obj.layer_num)