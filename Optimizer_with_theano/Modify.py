from .Layer import *


class Reshape(Layer):
    def __init__(self, obj, n_out, name=None):
        super().__init__(obj, name=name)
        self.n_out   = n_out

    def out(self):
        obj = self.obj
        obj.out = obj.out.reshape([-1, *self.n_out])
        return obj
    
    def gen_name(self):
        if self.name is None:
            self.name = "Reshape_{}".format(self.obj.layer_info.layer_num)

class Flatten(Layer):
    def __init__(self, obj, name=None):
        super().__init__(obj, name=name)

    def out(self):
        obj = self.obj
        self.n_out = (np.array(self.n_in).prod(),)
        obj.out = obj.out.reshape((-1, *self.n_out))
        return obj

    def gen_name(self):
        if self.name is None:
            self.name = "Flatten_{}".format(self.obj.layer_info.layer_num)
