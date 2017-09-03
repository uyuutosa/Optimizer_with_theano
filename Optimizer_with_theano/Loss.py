import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

_EPSILON = 10e-8


def mse(obj):
    diff = obj.y - obj.out
    obj.loss =  T.mean((diff) ** 2)
    
    # coefficient of determination
    obj.train_acc = 1 - ((diff**2).sum() / ((obj.y.flatten() - obj.y.mean())**2).sum())
    obj.valid_acc = 1 - ((diff**2).sum() / ((obj.y.flatten() - obj.y.mean())**2).sum())
    
    return obj
    
def mse_self(obj, input_tensor):
    diff = obj.y - obj.out
    obj.loss =  T.mean((input_tensor) ** 2)
    
    # coefficient of determination
    obj.train_acc = 1 - ((input_tensor**2).sum() / ((obj.y.flatten() - obj.y.mean())**2).sum())
    obj.valid_acc = 1 - ((input_tensor**2).sum() / ((obj.y.flatten() - obj.y.mean())**2).sum())
    
    return obj

def loss_self(obj, input_tensor, input_y=False):
    obj.loss = input_tensor
    
    # coefficient of determination
    obj.train_acc = theano.shared(0)
    obj.valid_acc = theano.shared(0)
    
    return obj


def cross_entropy(obj):
    obj.out = T.clip(obj.out, _EPSILON, 1.0 - _EPSILON)
    obj.loss =  nnet.categorical_crossentropy(obj.out, obj.y).mean()
    
    # one-hot to serial 
    obj.out  = obj.out.argmax(axis=1)[:,None]
    obj.y    = obj.y.argmax(axis=1)[:,None]
    
    # classification accuracy
    obj.train_acc = (T.eq(obj.out,obj.y).sum().astype(theano.config.floatX) / obj.n_batch)
    obj.valid_acc = (T.eq(obj.out,obj.y).sum().astype(theano.config.floatX) / obj.x_test_arr.shape[0])
    
    return obj

def binary_cross_entropy(obj):
    obj.out = T.clip(obj.out, _EPSILON, 1.0 - _EPSILON)
    obj.loss =  nnet.binary_crossentropy(obj.out, obj.y).mean()
    
    # classification accuracy
    diff = obj.y - obj.out
    #obj.train_acc = (T.eq(obj.out,obj.y).sum().astype(theano.config.floatX) / obj.n_batch)
    #obj.valid_acc = (T.eq(obj.out,obj.y).sum().astype(theano.config.floatX) / obj.x_test_arr.shape[0])
    obj.train_acc = 1 - ((diff**2).sum() / ((obj.y.flatten() - obj.y.mean())**2).sum())
    obj.valid_acc = 1 - ((diff**2).sum() / ((obj.y.flatten() - obj.y.mean())**2).sum())
    
    return obj