from .Layer import *

class Input_layer():
    def __init__(self, obj, name=None):
        self.obj = obj
        self.n_in = self.n_out = obj.x_train_arr.shape[1:]
        self.obj.params = []

    def gen_name(self):
        if self.name is None:
            self.name = "Input_{}".format(self.obj.layer_num)
    
