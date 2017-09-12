import theano as th
import theano.tensor as T


def Activation(name):
    if name == "sin":
        return T.sin
    elif name == "cos":
        return T.cos
    elif name == "tanh":
        return T.tanh
    elif name == "sigmoid":
        return T.nnet.sigmoid
    elif name == "relu":
        return T.nnet.relu
    elif name == "softmax":
        return T.nnet.softmax
    elif name == "linear" or name is None:
        return lambda x:x
        