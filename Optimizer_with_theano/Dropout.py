from .Layer import *
from theano.sandbox.rng_mrg import MRG_RandomStreams 

class Dropout(Layer):
    def __init__(self, obj, rate, seed=None, name=None):
        super().__init__(obj, name=name)
        self.srng = MRG_RandomStreams(12345 if seed is None else seed)
        self.rate = theano.shared(rate)#.astype(theano.config.floatX)
        self.obj.layer_info.set_dropout_param(rate, self.rate)
        #obj.dropout_rate_lst += [(rate, self.rate)]
        #self.n_out = obj.layerlst[-1].n_out

    def out(self):
        obj     = self.obj
        obj.out = T.where(self.srng.uniform(size=obj.out.shape) > self.rate, obj.out, 0)
        return obj

    def update(self):
        self.out()
        return self.obj
    
    def gen_name(self):
        if self.name is None:
            self.name = "Dropout_{}".format(self.obj.layer_info.layer_num)