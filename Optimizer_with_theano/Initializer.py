import numpy as np


class Initializer:
    def __init__(self, name="xavier", **kwarg):
        if name == "xavier":
            self.out =  self.xavier(**kwarg)
        elif name == "zeros":
            self.out = self.zeros(**kwarg)
        elif name == "ones":
            self.out = self.ones(**kwarg)
        elif name == "any":
            self.out = self.any(**kwarg)
            
    def xavier(self, **kwarg):
        if "n_in" in kwarg:
            n_in = np.array(kwarg["n_in"]).prod()
            n_out = np.array(kwarg["n_out"]).prod()
            num = n_in + n_out
        else:
            num = np.array(kwarg["shape"]).prod()
            
        v =  np.sqrt(2 / num) * \
               self.random_kinds(kwarg["random_kinds"])(*kwarg["random_params"], kwarg["shape"])
        return v
       
    def random_kinds(self, name):
        # https://docs.scipy.org/doc/numpy/reference/routines.random.html
        if name == "uniform":
            return np.random.uniform
        elif name == "normal":
            return np.random.normal
        elif name == "beta":
            return np.random.beta
        elif name == "exponential":
            return np.random.exponential
           
            
        
    