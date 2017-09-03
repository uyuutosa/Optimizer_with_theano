from .Layer import *
from .Util import conv_ndarr_if_shared

class Input_layer(Layer):
    def __init__(self, obj, input_shape=None, name=None):
#        super().__init__(obj, name=name)
        self.obj = obj
        if input_shape is None:
            self.n_in = self.n_out = conv_ndarr_if_shared(obj.x_train_arr).shape[1:]
        else:
            self.n_in = self.n_out = input_shape
        self.name = name
        self.gen_name()
        self.params = {}
        self.actname = None
        
    def update(self, obj=None):
        if obj is not None:
            self.obj = obj
        self.out()
        return self.obj

    def gen_name(self):
        if self.name is None:
            self.name = "Input_{}".format(self.obj.layer_info.layer_num)
    
