from .Initializer import Initializer
from .Layer import *


class Dense_layer(Layer):
    def __init__(self, 
                 obj, 
                 n_out, 
                 init_kinds="xavier", 
                 random_kinds="normal", 
                 random_params=(0, 1),
                 activation="linear",
                 is_train=True,
                 name=None
                ):
        super().__init__(obj, activation=activation, name=name)
        self.n_out   = (n_out,)
        self.theta   = theano.shared(Initializer(name=init_kinds,
                                                 n_in=self.n_in,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(*self.n_in, n_out),
                                                 ).out.astype(dtype=theano.config.floatX), borrow=True)
        
        self.b       = theano.shared(Initializer(name=init_kinds,
                                                 n_in=self.n_in,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(n_out),
                                                 ).out.astype(dtype=theano.config.floatX), borrow=True)
        
        self.gen_name()
        if is_train:
            self.params = {self.name + "_theta":self.theta, self.name + "_b":self.b}

    def out(self):
        obj     = self.obj
        obj.out = obj.out.dot(self.theta) + self.b

    
    def gen_name(self):
        if self.name is None:
            self.name = "Dense_{}".format(self.obj.layer_info.layer_num)

            
class Accum_layer(Layer):
    def __init__(self, 
                 obj, 
                 n_out, 
                 init_kinds="xavier", 
                 random_kinds="normal", 
                 random_params=(0, 1),
                 name=None
                ):
        super().__init__(obj, name=name)
        n_in = obj.layerlst[-1].n_out
        self.n_out   = (n_out,)
#        self.b       = theano.shared(Initializer(name=init_kinds,
#                                                 n_in=n_in,
#                                                 n_out=n_out,
#                                                 random_kinds=random_kinds,
#                                                 random_params=random_params,
#                                                 shape=(n_in)).out.astype(dtype=theano.config.floatX), borrow=True)
        self.theta   = theano.shared(Initializer(name=init_kinds,
                                                 n_in=n_in,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(*n_in, n_out),
                                                 ).out.astype(dtype=theano.config.floatX), borrow=True)
    
        self.b       = theano.shared(Initializer(name=init_kinds,
                                                 n_in=n_in,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=(n_out),
                                                 ).out.astype(dtype=theano.config.floatX), borrow=True)
        self.obj.params += [self.b, self.theta]

    def out(self):
        obj     = self.obj
        obj.out = ((obj.out[...,None] + self.b) + self.theta[None, ...]).sum(axis=1)

    def gen_name(self):
        if self.name is None:
            self.name = "Dense_{}".format(self.obj.layer_info.layer_num)