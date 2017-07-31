from Layer import *


class dense_layer(Layer):
    def __init__(self, obj, n_out, name=None):
        super().__init__(obj, name=name)
        self.n_out   = (n_out,)
        print(*self.n_in, n_out)
        self.theta   = theano.shared(np.random.rand(*self.n_in, n_out).astype(theano.config.floatX),
                                    borrow=True)
        #self.b       = theano.shared(np.random.rand(1).astype(dtype=theano.config.floatX)[0],
        self.b       = theano.shared(np.random.rand(n_out).astype(dtype=theano.config.floatX),
                                    borrow=True)
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