from .Initializer import Initializer
from .Layer import *


class Dense_layer(Layer):
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
        self.obj.params += [self.theta, self.b]

    def out(self):
        obj     = self.obj
        obj.out = obj.out.dot(self.theta) + self.b

    def update(self):
        self.out()
        self.obj.update_node([self.n_out])
        return self.obj
    
    def gen_name(self):
        if self.name is None:
            self.name = "Dense_{}".format(self.obj.layer_num)
