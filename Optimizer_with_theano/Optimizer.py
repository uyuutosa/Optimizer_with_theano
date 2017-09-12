import copy as cp
import os
import cv2
import scipy
from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from PIL import Image
#from sklearn.cross_validation import train_test_split
from sklearn.datasets import *
from sklearn.datasets import fetch_mldata
from theano.printing import pydotprint
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compile.nanguardmode import NanGuardMode
from theano.compile.debugmode import DebugMode
import pickle
import sys
sys.setrecursionlimit(10000)

from .Input import Input_layer
from .Dense import Dense_layer, Accum_layer
from .Polynominal import Polynominal_layer
from .Conv import Conv2D_layer
from .RNN  import RNN_layer
from .Pool import Pool_layer, Unpool_layer

from .Dropout import Dropout
from .Modify import Flatten, Reshape
from .Datasets import set_datasets
from .LayerInfo import Layer_info
from .Loss import mse, mse_self, cross_entropy, binary_cross_entropy, loss_self
from .Activation import *
from .CompileAndOptimize import *
from .Util import train_test_split

#import Datasets as ds

_EPSILON = 10e-8

theano.config.exception_verbosity = "high"

class optimizer:
    def __init__(self, 
                 x_arr=None, 
                 y_arr=None, 
                 out=None, 
                 thetalst=None, 
                 test_size=0.1, 
                 n_batch=500):
        
        self.n_batch = theano.shared(int(n_batch))
        
        if x_arr is not None and y_arr is not None:
            self.set_data(x_arr, y_arr, test_size)
            self.set_variables()
        
        self.thetalst = [] #if thetalst is None else thetalst
        self.dropout_rate_lst = []
        
        self.n_view = None
        self.updatelst = []
        self.tmplst = []
        self.layer_info = Layer_info()
    
    def set_data(self, 
                 x_arr, 
                 y_arr, 
                 test_size=0.1, 
                 input_shape=None,
                 is_shuffle=True):
        
        x_arr = x_arr.astype(theano.config.floatX)
        y_arr = y_arr.astype(theano.config.floatX)
        
        self.x_arr = x_arr
        self.y_arr = y_arr
        
        self.x_train_arr, \
        self.x_test_arr,\
        self.y_train_arr,\
        self.y_test_arr,\
        self.n_row,\
        self.train_n_row,\
        self.test_n_row\
        = train_test_split(x_arr, y_arr,
                           test_size=test_size,
                           is_shuffle=is_shuffle)
        
        self.set_variables()
        self.layer_info.set_layer(Input_layer(self, input_shape=input_shape))
        return self
    
    def set_layers(self, layerlst):
        #self.layer_info = Layer_info()
        self.layer_info.set_layers(layerlst)
        obj = self.layer_info.update_layers(self)
        return obj
        
    def set_variables(self, 
                      x_given=None,
                      y_given=None,):
        if self.n_batch.get_value() > self.n_row: 
            self.n_batch.set_value(int(self.n_row))
            
#        self.n_data = self.n_row
        n_xdim = self.x_train_arr.ndim
        n_ydim = self.y_train_arr.ndim
        
        if type(self.x_arr) is not np.ndarray:
            self.x = self.x_arr
        else:
            if  n_xdim == 0:
                self.x = T.scalar("x")
            if  n_xdim == 1:
                self.x_train_arr = self.x_train_arr[:,None]
                self.x_test_arr = self.x_test_arr[:,None]
                self.x = T.matrix("x")
            elif n_xdim == 2:
                self.x = T.matrix("x")
            elif n_xdim == 3:
                self.x = T.tensor3("x")
            else:
                self.x = T.tensor4("x")
            
        if type(self.y_arr) is not np.ndarray:
            self.y = self.y_arr
        else:
            if n_ydim == 0:
                self.y = T.scalar("y")
            if n_ydim == 1:
                self.y_train_arr = self.y_train_arr[:,None]
                self.y_test_arr = self.y_test_arr[:,None]
                self.y = T.matrix("y")
            elif n_ydim == 2:
                self.y = T.matrix("y")
            elif n_ydim == 3:
                self.y = T.tensor3("y")
            else:
                self.y = T.tensor4("y")
            
        self.out = self.x  #if out is None else out
        #self.batch_shape_of_C = T.concatenate([T.as_tensor([self.n_batch]), theano.shared(np.array([3]))], axis=0)
        if x_given is None:
            self.xlst = [self.x]
        else:
            self.xlst = [x_given]
        if y_given is None:
            self.ylst = [self.y]
        else:
            self.ylst = [y_given]
            
        self.train_xgivenlst = [self.x_train_arr]
        self.train_ygivenlst = [self.y_train_arr]
        self.test_xgivenlst  = [self.x_test_arr]
        self.test_ygivenlst  = [self.y_test_arr]
        
        
    def set_datasets(self, 
                     data="mnist", 
                     is_one_hot=True, 
                     test_size=0.1, 
                     is_shuffle=False, 
                     **kwarg):
        obj = self.copy()
        obj = obj.set_data(*set_datasets(data, is_one_hot, **kwarg), test_size=test_size, is_shuffle=is_shuffle)
        return obj
    
    def copy(self):
        return self
        #return cp.copy(self)
        
    
        
    
    def dropout(self, rate=0.5, seed=None, name=None):
        obj = self.copy()
        layer = Dropout(obj, 
                        rate,
                        seed,
                        name
                        )
        
        obj   = layer.update()
        return obj
        
    def dense(self, 
              n_out,
              init_kinds="xavier",
              random_kinds="normal",
              random_params=(0, 1),
              act="linear",
              is_train=True,
              name=None
             ):
        obj = self.copy()
        layer = Dense_layer(obj, 
                            n_out,
                            init_kinds,
                            random_kinds,
                            random_params,
                            activation=act,
                            is_train=is_train,
                            name=name
                           )
        obj   = layer.update()
        return obj
    
    def accum(self, 
              n_out,
              init_kinds="xavier",
              random_kinds="normal",
              random_params=(0, 1),
              act="linear",
              name=None
             ):
        obj = self.copy()
        layer = Accum_layer(obj, 
                            n_out,
                            init_kinds,
                            random_kinds,
                            random_params
                           )
        obj   = layer.update()
        return obj
    
    def rnn(self, 
            n_out, 
            axis=0, 
            init_kinds="xavier",
            random_kinds="normal",
            random_params=(0, 1),
            is_out=False,
            act="linear",
            name=None,
           ):
        obj = self.copy()
        layer = RNN_layer(obj,
                          n_out, 
                          axis, 
                          init_kinds,
                          random_kinds,
                          random_params,
                          is_out,
                          name,
                          activation
                          )
        obj   = layer.update()
        return obj
    

    
    def conv2d(self, 
               kshape=(1,1,3,3), 
               mode="full", 
               reshape=None,
               init_kinds="xavier",
               random_kinds="normal",
               random_params=(0, 1),
               act="linear",
               theta=None,
               b=None,
               is_train=True,
               name=None
              ):
        obj = self.copy()
        layer = Conv2D_layer(obj, 
                             kshape, 
                             mode, 
                             reshape,
                             init_kinds,
                             random_kinds,
                             random_params,
                             act,
                             theta,
                             b,
                             is_train,
                             name
                            )
        obj   = layer.update()
        return obj
    
    def conv_and_pool(self, 
                      fnum, 
                      height, 
                      width, 
                      mode="full", 
                      act="linear",
                      ds=(2,2)):
        obj = self.copy()
        kshape = (fnum, height, width)
        obj = obj.conv2d(kshape=kshape, mode=mode, act=act)
        obj = obj.pool(ds=ds)
        n_in = obj.layer_info.get_shape_of_last_node()
        if mode == "full":
            #n_in = obj.get_curr_node()
            obj = obj.reshape((kshape[0], *n_in[-2:]))
            #obj = obj.reshape((kshape[0], M[0]+(m[0]-1),M[1]+(m[1]-1)))
        elif mode == "valid":
            #n_in = obj.get_curr_node()
            obj = obj.reshape((kshape[0], *n_in[-2:]))
        return obj
    
    def pool(self, 
             ds=(2,2),
             act="linear",
             name=None):
        obj = self.copy()
        layer = Pool_layer(obj, ds, activation=act, name=name)
        obj   = layer.update()
        return obj
    
    def unpool(self, 
             ds=(2,2),
             act="linear",
             name=None):
        obj = self.copy()
        layer = Unpool_layer(obj, ds, activation=act, name=name)
        obj   = layer.update()
        return obj
    
#    def mean(self, axis):
#        obj = self.copy()
#        
#        n_in = obj.get_curr_node()
#        obj.out = obj.out.mean(axis=axis)
#        obj.update_node(np.ones(n_in).mean(axis=axis).shape)
#        return obj
    
    def reshape(self, shape):
        obj = self.copy()
        layer = Reshape(obj, shape)
        obj   = layer.update()
        return obj

#    def reshape_(self, shape):
#        obj = self.copy()
#        obj.out = obj.out.reshape(shape)
#        obj.update_node(shape[1:])
#        return obj
    
    def flatten(self):
        obj = self.copy()
        layer = Flatten(obj)
        obj   = layer.update()
        return obj
    
    def poly(self, 
              M,
              n_out,
              init_kinds="xavier",
              random_kinds="normal",
              random_params=(0, 1),
              act="linear",
              name=None
             ):
        obj = self.copy()
        layer = Polynominal_layer(
                            obj, 
                            M,
                            n_out,
                            init_kinds,
                            random_kinds,
                            random_params,
                            act,
                            name
                           )
        obj   = layer.update()
        return obj
        
    def relu(self, ):
        obj     = self.copy()
        obj.out = nnet.relu(obj.out)
        return obj
    
    def tanh(self, ):
        obj = self.copy()
        obj.out = T.tanh(obj.out)
        return obj
    
    def sigmoid(self, ):
        obj = self.copy()
        obj.out = nnet.sigmoid(obj.out)
        return obj
    
    def softmax(self, ):
        obj = self.copy()
        obj.out = nnet.softmax(obj.out)
        return obj
        
    def loss_mse(self):
        obj = self.copy()
        obj = mse(obj)
        return obj
    
    def loss_mse_self(self, input_tensor):
        obj = self.copy()
        obj = mse_self(obj, input_tensor)
        return obj
   
    def loss_self(self, input_tensor):
        obj = self.copy()
        obj = loss_self(obj, input_tensor)
        return obj
    
    def loss_cross_entropy(self):
        obj = self.copy()
        obj = cross_entropy(obj)
        return obj
    
    def loss_binary_cross_entropy(self):
        obj = self.copy()
        obj = binary_cross_entropy(obj)
        return obj
    
    def opt_sgd(self, alpha=0.1, input_grads=[]):
        obj = self.copy()
        obj.updatelst = []
        params = obj.layer_info.get_params() + input_grads
        for theta in params:
            obj.updatelst += [(theta, theta - (alpha * T.grad(obj.loss, wrt=theta)))]
            
        return obj
    
    def opt_newton(self, alpha=0.1, input_grads=[]):
        obj = self.copy()
        obj.updatelst = [] 
        params = obj.layer_info.get_params() + input_grads
        for theta in params:
            g = T.grad(obj.loss, theta)
            H = theano.gradient.jacobian(g.flatten(), theta)
            obj.updatelst += [(theta, theta - (alpha * T.nlinalg.matrix_inverse(H).dot(T.grad(obj.loss, wrt=theta))))]
        return obj
    
    #def opt_bfgs(self, alpha=0.1, input_grads=[]):
    #    obj = self.copy()
    #    obj.updatelst = [] 
    #    params = obj.layer_info.get_params() + input_grads
    #    for theta in params:
    #        n = theta.flatten().shape[0]
    #        B = T.zeros((n, n))
    #        g = T.grad(obj.loss, theta)
    #        
    #        obj.updatelst += [(theta, theta - (alpha * T.nlinalg.matrix_inverse(g).dot(T.grad(obj.loss, wrt=theta))))]
    #    return obj
    
    def opt_RMSProp(self, alpha=0.001, gamma=0.9, ep=1e-8, input_grads=[]):
        obj = self.copy()
        obj.updatelst = []
        params = obj.layer_info.get_params() + input_grads
        rlst = [theano.shared(np.zeros(x.get_value().shape, theano.config.floatX)) for x in params]
        
        for r, theta in zip(rlst, params):
            g = T.grad(obj.loss, wrt=theta)
            obj.updatelst += [(r,     gamma * r + (1 - gamma) * g ** 2),\
                              (theta, theta - (alpha / (T.sqrt(r) + ep)) * g)]
            
        return obj
                               
    def opt_AdaGrad(self, ini_eta=0.001, ep=1e-8, input_grads=[]):
        obj = self.copy()
        obj.updatelst = []
        params = obj.layer_info.get_params() + input_grads
            
        hlst   = [theano.shared(ep*np.ones(x.get_value().shape, theano.config.floatX)) for x in params]
        etalst = [theano.shared(ini_eta*np.ones(x.get_value().shape, theano.config.floatX)) for x in params]
        
        for h, eta, theta in zip(hlst, etalst, params):
            g   = T.grad(obj.loss, wrt=theta)
            obj.updatelst += [(h,     h + g ** 2),
                              (eta,   eta / T.sqrt(h+1e-4)),
                              (theta, theta - eta * g)]
            
        return obj
    
    def opt_Adam(self, 
                 alpha=0.001, 
                 beta=0.9, 
                 gamma=0.999, 
                 ep=1e-8, 
                 t=3, 
                 input_grads=[]):
        
        obj = self.copy()
        obj.updatelst = []
        params = obj.layer_info.get_params() + input_grads
        nulst = [theano.shared(np.zeros(x.get_value().shape, theano.config.floatX)) for x in params]
        rlst = [theano.shared(np.zeros(x.get_value().shape, theano.config.floatX)) for x in params]
        
        for nu, r, theta in zip(nulst, rlst, params):
            g = T.grad(obj.loss, wrt=theta)
            nu_hat = nu / (1 - beta)
            r_hat = r / (1 - gamma)
            obj.updatelst += [(nu, beta * nu + (1 - beta) * g),\
                              (r, gamma * r +  (1 - gamma) * g ** 2),\
                              (theta, theta - alpha*(nu_hat / (T.sqrt(r_hat) + ep)))]
            
        return obj
    
    
    def compile(self, is_fast=True, is_random=True):
        
        if is_fast:
            self.CO = CO_fast_but_takes_a_lot_of_memory(self, is_random)
        else:
            self.CO = CO_slow_but_only_few_memory_needed(self, is_random)
        
        return self.CO.compile()
    
            
    def optimize(self, 
                 n_epoch=100, 
                 n_view=10, 
                 n_iter=None, 
                 n_batch=None, 
                 is_valid=True,
                 is_view=True):
        
        return self.CO.optimize(n_epoch, n_view, n_iter, n_batch, is_valid, is_view)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.copy(), f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

        
    def view(self, yscale="log", is_valid=True):
        if not len(self.train_loss_lst):
            raise ValueError("Loss value is not be set.")
        plt.clf()

        train_idx = np.linspace(0, self.n_epoch, len(self.train_loss_lst))
        valid_idx = np.arange(self.n_epoch)
        
        plt.subplot(2,1,1)
        plt.ylabel("Loss")
        plt.yscale(yscale)
        plt.plot(train_idx, self.train_loss_lst, c="r", label="train")
        if is_valid:
            plt.plot(valid_idx, self.valid_loss_lst, c="b", label="validate")
        plt.legend()
        

        plt.subplot(2,1,2)
        plt.ylim(0, 1.1)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(train_idx, self.train_acc_lst, c="r", label="train")
        if is_valid:
            plt.plot(valid_idx, self.valid_acc_lst, c="b", label="validate")
        plt.legend()
        plt.show()
    
    def view_graph(self, width='100%', res=60):
        path = 'examples'; name = 'mlp.png'
        path_name = path + '/' + name 
        if not os.path.exists(path):
            os.makedirs(path)
        pydotprint(self.loss, path_name)
        plt.figure(figsize=(res, res), dpi=80)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, hspace=0.0, wspace=0.0)
        plt.axis('off')
        plt.imshow(np.array(Image.open(path_name)))
        plt.show()
    
    
        self.n_batch.set_value(int(x_arr.shape[0]))
        return self.pred_func(x_arr) 
        #print(self.h.get_value())
    
    def __add__(self, other):
        obj = self.copy()
        obj.out += other.out
        if self.x is not other.x:
            obj.xlst += other.xlst
            obj.ylst += other.ylst
            obj.train_xgivenlst += other.train_xgivenlst
            obj.test_xgivenlst += other.test_xgivenlst
            obj.train_ygivenlst += other.train_ygivenlst
            obj.test_ygivenlst += other.test_ygivenlst
        return obj
    
    def __sub__(self, other):
        obj = self.copy()
        obj.out -= other.out
        if self.x is not other.x:
            obj.xlst += other.xlst
            obj.ylst += other.ylst
            obj.train_xgivenlst += other.train_xgivenlst
            obj.test_xgivenlst += other.test_xgivenlst
            obj.train_ygivenlst += other.train_ygivenlst
            obj.test_ygivenlst += other.test_ygivenlst
        return obj
    
    def __mul__(self, other):
        obj = self.copy()
        obj.out *= other.out
        if self.x is not other.x:
            obj.xlst += other.xlst
            obj.ylst += other.ylst
            obj.train_xgivenlst += other.train_xgivenlst
            obj.test_xgivenlst  += other.test_xgivenlst
            obj.train_ygivenlst += other.train_ygivenlst
            obj.test_ygivenlst  += other.test_ygivenlst
        return obj
    
    def __truediv__(self, other):
        obj = self.copy()
        obj.out /= other.out
        if self.x is not other.x:
            obj.xlst += other.xlst
            obj.ylst += other.ylst
            obj.train_xgivenlst += other.train_xgivenlst
            obj.test_xgivenlst  += other.test_xgivenlst
            obj.train_ygivenlst += other.train_ygivenlst
            obj.test_ygivenlst  += other.test_ygivenlst
        return obj
    
    def __pow__(self, other):
        obj = self.copy()
        obj.out **= other.out
        if self.x is not other.x:
            obj.xlst += other.xlst
            obj.ylst += other.ylst
            obj.train_xgivenlst += other.train_xgivenlst
            obj.test_xgivenlst  += other.test_xgivenlst
            obj.train_ygivenlst += other.train_ygivenlst
            obj.test_ygivenlst  += other.test_ygivenlst
        return obj
    
        
