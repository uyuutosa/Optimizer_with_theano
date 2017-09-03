from .Initializer import Initializer
from .Layer import *
from .Activation import *


class Polynominal_layer(Layer):
    def __init__(self, 
                 obj, 
                 M,
                 n_out, 
                 init_kinds="xavier", 
                 random_kinds="normal", 
                 random_params=(0, 1),
                 activation="linear",
                 name=None
                ):
        super().__init__(obj, activation=activation, name=name)
        n_in = obj.layer_info.layerlst[-1].n_out[0]
        self.n_out   = (n_out,)
        self.obj = obj
        
        n_batch = obj.out.shape[0]
        x_times = T.concatenate([obj.out, T.ones((n_batch, 1)).astype(theano.config.floatX)],axis=1)
        shapelst = [n_batch, n_in+1]
        for i in range(M-1):
            shapelst += [n_in+1]
            x_times = Activation(activation)((x_times[...,None] * T.ones(shapelst).astype(theano.config.floatX)) * x_times[..., None])

        self.x_times = x_times.reshape((n_batch, -1)) 
        
        
        self.theta   = theano.shared(Initializer(name=init_kinds,
                                                 n_in=(n_in+1) ** M,
                                                 n_out=n_out,
                                                 random_kinds=random_kinds,
                                                 random_params=random_params,
                                                 shape=((n_in+1) ** M, n_out),
                                                 ).out.astype(dtype=theano.config.floatX) * M ** (-M), borrow=True) 
        self.obj.params += [self.theta]

    def out(self):
        obj     = self.obj
        obj.out = self.x_times.dot(self.theta.astype(theano.config.floatX))

    def update(self):
        self.out()
        self.obj.update_node([self.n_out])
        return self.obj
    
    def gen_name(self):
        if self.name is None:
            self.name = "Polynominal_{}".format(self.obj.layer_num)
