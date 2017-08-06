from .Layer import *


class Dense_layer(Layer):
    def __init__(self, obj, n_out, name=None):
        """Provides a unique UID given a string prefix.
            # Arguments
              prefix: string.
            # Returns
              An integer.
            # Example
            ```python
                >>> keras.backend.get_uid('dense')
                1
                >>> keras.backend.get_uid('dense')
                2
            ```
        """
        super().__init__(obj, name=name)
        self.n_out   = (n_out,)
        self.theta   = theano.shared(np.random.rand(*self.n_in, n_out).astype(theano.config.floatX),
                                    borrow=True)
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
