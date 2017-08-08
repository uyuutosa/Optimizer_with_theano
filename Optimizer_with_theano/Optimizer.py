import copy as cp
import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from PIL import Image
from sklearn.cross_validation import train_test_split
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
from .Dense import Dense_layer
from .Polynominal import Polynominal_layer
from .Conv import Conv2D_layer
from .RNN  import RNN_layer

from .Pool import Pool_layer
from .Modify import Flatten, Reshape
from .Datasets import set_datasets
#import Datasets as ds

_EPSILON = 10e-8

theano.config.exception_verbosity = "high"


#mpl.rc("savefig", dpi=1200)

#%config InlineBackend.rc = {'font.size': 20, 'figure.figsize': (12.0, 8.0),'figure.facecolor': 'white', 'savefig.dpi': 72, 'figure.subplot.bottom': 0.125, 'figure.edgecolor': 'white'}
#%matplotlib inline

class optimizer:
    def __init__(self, 
                 x_arr=None, 
                 y_arr=None, 
                 out=None, 
                 thetalst=None, 
                 nodelst=None,
                 test_size=0.1, 
                 n_batch=500):
        
        self.n_batch = theano.shared(int(n_batch))
        
        if x_arr is not None and y_arr is not None:
            self.set_data(x_arr, y_arr, test_size)
            self.set_variables()
        
        self.thetalst = [] #if thetalst is None else thetalst
        
        self.n_view = None
        self.updatelst = []
        self.tmplst = []
        self.layer_num = 0
    
    def set_data(self, x_arr, y_arr, test_size=0.1, is_shuffle=False):
        
        x_arr = x_arr.astype(theano.config.floatX)
        y_arr = y_arr.astype(theano.config.floatX)
        
        if is_shuffle:
            self.x_train_arr, \
            self.x_test_arr,\
            self.y_train_arr,\
            self.y_test_arr,\
            = train_test_split(x_arr, y_arr,
                               test_size = test_size)
        else:
            lim = int(x_arr.shape[0] * (1 - test_size))
            self.x_train_arr = x_arr[:lim]
            self.x_test_arr  = x_arr[lim:]
            self.y_train_arr = y_arr[:lim]
            self.y_test_arr  = y_arr[lim:]
        
        self.nodelst = [[int(np.prod(self.x_train_arr.shape[1:]))]] # if nodelst is None else nodelst
        self.layerlst = [Input_layer(self)]
        
        self.train_xgivenlst = [self.x_train_arr]
        self.train_ygivenlst = [self.y_train_arr]
        self.test_xgivenlst = [self.x_test_arr]
        self.test_ygivenlst = [self.y_test_arr]
        
    
    def set_variables(self):
        if self.n_batch.get_value() > self.x_train_arr.shape[0]: 
            self.n_batch.set_value(int(self.x_train_arr.shape[0]))
        self.n_data = self.x_train_arr.shape[0]
        n_xdim = self.x_train_arr.ndim
        n_ydim = self.y_train_arr.ndim
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
        self.xlst = [self.x]
        self.ylst = [self.y]
        
    def set_datasets(self, data="mnist", is_one_hot=True):
        obj = self.copy()
        obj.set_data(*set_datasets(data, is_one_hot))
        obj.set_variables()
        return obj
    
    def copy(self):
        return self
        #return cp.copy(self)
    
    def update_node(self, n_out):
        self.nodelst = self.nodelst + [n_out]
        
    def get_curr_node(self):
        return list(self.nodelst[-1])
    
    #def dropout(self, rate=0.5, seed=None):
    #    obj = self.copy()
    #    
    #    srng = RandomStreams(seed)
    #    obj.out = T.where(srng.uniform(size=obj.out.shape) > rate, obj.out, 0)
    #    
    #    return obj
        
    def dense(self, 
              n_out,
              init_kinds="xavier",
              random_kinds="normal",
              random_params=(0, 1),
              name=None
             ):
        obj = self.copy()
        layer = Dense_layer(obj, 
                            n_out,
                            init_kinds,
                            random_kinds,
                            random_params
                           )
        obj   = layer.update()
        obj.layerlst += [layer]
        return obj
    
    def rnn(self, 
            n_out, 
            axis=0, 
            init_kinds="xavier",
            random_kinds="normal",
            random_params=(0, 1),
            is_out=True,
            name=None,
            activation="relu",
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
        obj.layerlst += [layer]
        return obj
    
    #def lstm(self):
    #    obj = self.copy()
    #    
    #    curr_shape = obj.get_curr_node()
    #    n_in = n_out = curr_shape[-1]
    #    
    #    #batch_shape_of_h = T.concatenate(
    #    #batch_shape_of_C = T.concatenate(, axis=0)
#   #     h = T.ones(theano.sharedl())
    #    h = T.zeros([obj.n_batch, *curr_shape], dtype=theano.config.floatX)
    #    C = T.zeros([obj.n_batch, n_out], dtype=theano.config.floatX)
    #    
    #    Wi =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
    #    Wf =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
    #    Wc =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
    #    Wo =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
    #    bi =  theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
    #    bf =  theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
    #    bc =  theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
    #    bo =  theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
    #    Ui =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
    #    Uf =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
    #    Uc =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
    #    Uo =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
    #    
    #    i = nnet.sigmoid(obj.out.dot(Wi) + h.dot(Ui) + bi)
    #    
    #    C_tilde = T.tanh(obj.out.dot(Wc) + h.dot(Uc) + bc)
    #    
    #    f = nnet.sigmoid(obj.out.dot(Wf) + h.dot(Uf) + bf)
    #    
    #    tmp = (i * C_tilde + f * C).reshape(C.shape)
    #    
    #    obj.tmplst += [(C, tmp)]
    #    
    #    C = tmp
    #    
    #    o = nnet.sigmoid(obj.out.dot(Wo) + h.dot(Uo) + bo)
    #    
    #    tmp = (o * T.tanh(C)).reshape(h.shape)
    #    
    #    obj.tmplst += [(h, tmp)]
    #    
    #    obj.out =  tmp
    #    
    #    obj.thetalst += [Wi, bi, Ui, Wf, bf, Uf, Wc, bc, Uc, Wo, bo, Uo]
    #    
    #    obj.update_node([n_out])
    #    
    #    return obj

    
    def conv2d(self, 
               kshape=(1,1,3,3), 
               mode="full", 
               reshape=None,
               init_kinds="xavier",
               random_kinds="normal",
               random_params=(0, 1),
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
                             name
                            )
        obj   = layer.update()
        obj.layerlst += [layer]
        return obj
    
    def conv_and_pool(self, 
                      fnum, 
                      height, 
                      width, 
                      mode="full", 
                      ds=(2,2)):
        obj = self.copy()
        n_in = obj.layerlst[-1].n_out
        #n_in = obj.get_curr_node()
        kshape = (fnum, n_in[0], height, width)
        n_batch = obj.n_batch.get_value()
        obj = obj.conv2d(kshape=kshape, mode=mode)
        #.reshape((n_batch, np.array(n_in, dtype=int).sum()))\
#
        obj = obj.relu().pool(ds=ds)
        if mode == "full":
            n_in = obj.get_curr_node()
            obj = obj.reshape((kshape[0], *n_in[-2:]))
            #obj = obj.reshape((kshape[0], M[0]+(m[0]-1),M[1]+(m[1]-1)))
        elif mode == "valid":
            n_in = obj.get_curr_node()
            obj = obj.reshape((kshape[0], *n_in[-2:]))
        return obj
            
    
    def pool(self, ds=(2,2)):
        obj = self.copy()
        layer = Pool_layer(obj, ds)
        obj   = layer.update()
        obj.layerlst += [layer]
        return obj
    
    def mean(self, axis):
        obj = self.copy()
        
        n_in = obj.get_curr_node()
        obj.out = obj.out.mean(axis=axis)
        obj.update_node(np.ones(n_in).mean(axis=axis).shape)
        return obj
    
    def reshape(self, shape):
        obj = self.copy()
        layer = Reshape(obj, shape)
        obj   = layer.update()
        obj.layerlst += [layer]
        return obj

    def reshape_(self, shape):
        obj = self.copy()
        obj.out = obj.out.reshape(shape)
        obj.update_node(shape[1:])
        return obj
    
    def flatten(self):
        obj = self.copy()
        layer = Flatten(obj)
        obj   = layer.update()
        obj.layerlst += [layer]
        return obj
    
    def poly(self, 
              M,
              n_out,
              init_kinds="xavier",
              random_kinds="normal",
              random_params=(0, 1),
              activation="linear",
              name=None
             ):
        obj = self.copy()
        layer = Polynominal_layer(obj, 
                            M,
                            n_out,
                            init_kinds,
                            random_kinds,
                            random_params,
                            activation,
                            name
                           )
        obj   = layer.update()
        obj.layerlst += [layer]
        return obj
        #obj = self.copy()
        #n_in = int(np.asarray(obj.get_curr_node()).sum())
        #
        #x_times = T.concatenate([obj.out, T.ones((obj.n_batch, 1)).astype(theano.config.floatX)],axis=1)
        #for i in range(M-1):
        #    idx = np.array([":,"]).repeat(i+1).tolist()
        #    a = dict()
        #    exec ("x_times = x_times[:," + "".join(idx) + "None] * x_times", locals(), a)
        #    x_times = a["x_times"]
#
        #x_times = x_times.reshape((obj.n_batch, -1))
#
        #theta = theano.shared(np.ones(((n_in+1) ** M, n_out)).astype(theano.config.floatX))
        #obj.theta = theano.shared(np.ones(((n_in+1) ** M, n_out)).astype(theano.config.floatX))
        #
        #obj.out = x_times.dot(theta.astype(theano.config.floatX))
        #obj.thetalst += [theta]
        #
        #obj.update_node([n_out])
        #
        #return obj

        
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
        obj.loss =  T.mean((obj.out - obj.y) ** 2)
        return obj
    
    def loss_mse_self(self, input_tensor):
        obj = self.copy()
        obj.loss =  T.mean((input_tensor) ** 2)
        return obj
    
    def loss_cross_entropy(self):
        obj = self.copy()
        obj.out = T.clip(obj.out, _EPSILON, 1.0 - _EPSILON)
        obj.loss =  nnet.categorical_crossentropy(obj.out, obj.y).mean()
        #obj.loss =  T.mean(nnet.categorical_crossentropy(obj.out, obj.y))
        #obj.loss =  -T.mean(obj.y * T.log(obj.out + 1e-8) -(1-obj.y) * T.log(1-obj.out + 1e-8))
        obj.out  = obj.out.argmax(axis=1)[:,None]
        obj.y    = obj.y.argmax(axis=1)[:,None]
        obj.params = obj.params[::-1]
        return obj
    
    def loss_softmax_cross_entropy(self):
        obj = self.copy()
        obj.out = nnet.softmax(obj.out)
        tmp_y = T.cast(obj.y, "int32")
        obj.loss = -T.mean(T.log(obj.out)[T.arange(obj.y.shape[0]), tmp_y.flatten()])
        obj.out  = obj.out.argmax(axis=1)[:,None]
        
        return obj
    
    def opt_sgd(self, alpha=0.1):
        obj = self.copy()
        obj.updatelst = []
        for theta in obj.params:
            obj.updatelst += [(theta, theta - (alpha * T.grad(obj.loss, wrt=theta)))]
            
        obj.updatelst += obj.tmplst
        return obj
    
    def opt_RMSProp(self, alpha=0.1, gamma=0.9, ep=1e-8):
        obj = self.copy()
        obj.updatelst = []
        obj.rlst = [theano.shared(x.shape).astype(theano.config.floatX) for x in obj.thetalst]
        
        for r, theta in zip(obj.rlst, obj.params):
            g = T.grad(obj.loss, wrt=theta)
            obj.updatelst += [(r, gamma * r + (1 - gamma) * g ** 2),\
                              (theta, theta - (alpha / (T.sqrt(r) + ep)) * g)]
        obj.updatelst += obj.tmplst
        return obj
                               
    def opt_AdaGrad(self, ini_eta=0.001, ep=1e-8):
        obj = self.copy()
        obj.updatelst = []
                               
        obj.hlst = [theano.shared(ep*np.ones(x.get_value().shape, theano.config.floatX)) for x in obj.thetalst]
        obj.etalst = [theano.shared(ini_eta*np.ones(x.get_value().shape, theano.config.floatX)) for x in obj.thetalst]
        
        for h, eta, theta in zip(obj.hlst, obj.etalst, obj.params):
            g   = T.grad(obj.loss, wrt=theta)
            obj.updatelst += [(h,     h + g ** 2),
                              (eta,   eta / T.sqrt(h+1e-4)),
                              (theta, theta - eta * g)]
            
        obj.updatelst += obj.tmplst
        return obj
    
    def opt_Adam(self, alpha=0.001, beta=0.9, gamma=0.999, ep=1e-8, t=3):
        obj = self.copy()
        obj.updatelst = []
        obj.nulst = [theano.shared(np.zeros(x.get_value().shape, theano.config.floatX)) for x in obj.params]
        obj.rlst = [theano.shared(np.zeros(x.get_value().shape, theano.config.floatX)) for x in obj.params]
        
        for nu, r, theta in zip(obj.nulst, obj.rlst, obj.params):
            g = T.grad(obj.loss, wrt=theta)
            nu_hat = nu / (1 - beta)
            r_hat = r / (1 - gamma)
            obj.updatelst += [(nu, beta * nu + (1 - beta) * g),\
                              (r, gamma * r +  (1 - gamma) * g ** 2),\
                              (theta, theta - alpha*(nu_hat / (T.sqrt(r_hat) + ep)))]
            
        obj.updatelst += obj.tmplst
        return obj
                               
    
    
    def compile(self, is_random=True):
        obj = self.copy()
        obj.dsize = obj.x_train_arr.shape[0]

        i = theano.shared(0).astype("int32")
        
        if is_random:
            obj.idx = theano.shared(np.random.permutation(obj.x_train_arr.shape[0]))
        else:
            obj.idx = theano.shared(np.arange(obj.x_train_arr.shape[0]))
            
        train_xgivens = []
        train_xgivens_acc = []
        for t, train_xgiven in zip(obj.xlst, obj.train_xgivenlst):
            xgiven_shared = theano.shared(train_xgiven).astype(theano.config.floatX)
            train_xgivens += [(t, xgiven_shared[obj.idx[i:obj.n_batch+i],])]
            train_xgivens_acc += [(t, xgiven_shared)]
        train_ygivens = []
        train_ygivens_acc = []
        for t, train_ygiven in zip(obj.ylst, obj.train_ygivenlst):
            ygiven_shared = theano.shared(train_ygiven).astype(theano.config.floatX)
            train_ygivens += [(t, ygiven_shared[obj.idx[i:obj.n_batch+i],])]
            train_ygivens_acc += [(t, ygiven_shared)]
            
        test_xgivens = []
        test_xgivens_acc = []
        for t, test_xgiven in zip(obj.xlst, obj.test_xgivenlst):
            xgiven_shared = theano.shared(test_xgiven).astype(theano.config.floatX)
            test_xgivens += [(t, xgiven_shared[obj.idx[i:obj.n_batch+i],])]
            test_xgivens_acc += [(t, xgiven_shared)]
            
        test_ygivens = []
        test_ygivens_acc = []
        for t, test_ygiven in zip(obj.ylst, obj.test_ygivenlst):
            ygiven_shared = theano.shared(test_ygiven).astype(theano.config.floatX)
            test_ygivens += [(t, ygiven_shared[obj.idx[i:obj.n_batch+i],])]
            test_ygivens_acc += [(t, ygiven_shared)]
            
        
        train_acc = (T.eq(obj.out,obj.y).sum().astype(theano.config.floatX) / obj.n_batch)
        obj.train_loss_and_acc = theano.function(inputs=[i],
                                        outputs=[obj.loss, train_acc],
                                        givens=train_xgivens+train_ygivens,
                                        #givens=[(obj.x,x_train_arr_shared[obj.idx[i:obj.n_batch+i],]),
                                        #        (obj.y,y_train_arr_shared[obj.idx[i:obj.n_batch+i],])],
                                        updates=obj.updatelst,
                                        #mode=DebugMode(check_c_code=False),
                                        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True), 
                                        on_unused_input='ignore')
        
        
        valid_acc = (T.eq(obj.out,obj.y).sum().astype(theano.config.floatX) / obj.x_test_arr.shape[0])
        obj.valid_loss_and_acc = theano.function(inputs=[],
                                        outputs=[obj.loss, valid_acc],
                                        givens=test_xgivens_acc+test_ygivens_acc,
                                        on_unused_input='ignore')
        
        
        
        obj.pred_func = theano.function(inputs  = obj.xlst,#[obj.x],
                                        outputs = obj.out,
                                        updates = obj.tmplst,
                                        on_unused_input='ignore')
    
        return obj
            
    def optimize(self, n_epoch=10, n_view=1000, n_iter=None):
        obj = self.copy()
        
        if obj.n_view is None: obj.n_view = n_view  
        obj.train_loss_lst = []
        obj.train_acc_lst  = []
        obj.valid_loss_lst = []
        obj.valid_acc_lst  = []
        try:
            obj.n_epoch = n_epoch
            for epoch in range(n_epoch):
                #obj.idx.set_value(np.random.permutation(obj.x_train_arr.shape[0]))
                #img = self.params[0]
                #s = img.size
                #cv2.imshow(img.reshape(np.sqrt(s), np.sqrt(s)))
                #cv2.waitkey(1)
                
                #for i in range (3):
                #    img = np.array(self.params[i].eval())
                #    s = img.size
                #    img = img.reshape((int(np.sqrt(s)), int(np.sqrt(s)))) * 255 - 124
                #    img = cv2.resize(img, (1000, 1000)).astype(np.uint8)
                #    cv2.imshow("hello{}".format(i), img)
                #cv2.waitKey(0)
                
                # update randomized index
                tmp = np.random.permutation(obj.x_train_arr.shape[0])
                obj.idx.set_value(tmp)
                
                mean_loss = 0.
                N = obj.dsize-obj.n_batch.get_value() + 1
                if n_iter is None:
                    n_iter = obj.n_batch.get_value()
                train_loss_lst = []
                train_acc_lst = []
                for i in range(0, N, n_iter):
                    train_loss_value, train_acc_value = obj.train_loss_and_acc(i)
                    train_loss_lst += [train_loss_value]
                    train_acc_lst  += [train_acc_value]
                valid_loss_value, valid_acc_value = obj.valid_loss_and_acc()
                train_mean_loss_value = np.array(train_loss_lst).mean()
                train_mean_acc_value  = np.array(train_acc_lst).mean()
                obj.train_loss_lst   += train_loss_lst
                obj.train_acc_lst    += train_acc_lst
                obj.valid_loss_lst   += [valid_loss_value]
                obj.valid_acc_lst    += [valid_acc_value]
                if not (epoch % n_view):
                    print("Epoch. %s: loss = %.4e, acc = %.4e, valid. loss = %.4e, valid. acc. = %.4e." %(epoch, 
                                                                                                        train_mean_loss_value, 
                                                                                                        train_mean_acc_value, 
                                                                                                        valid_loss_value,
                                                                                                        valid_acc_value))
            
        except KeyboardInterrupt:
            print ( "KeyboardInterrupt\n" )
            obj.n_epoch = epoch
            return obj
        return obj
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.copy(), f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    
    def view_params(self):
        for theta in self.thetalst:
            print(theta.get_value().shape)
            print(theta.get_value())
            
        
    def view(self, yscale="log"):
        if not len(self.train_loss_lst):
            raise ValueError("Loss value is not be set.")
        plt.clf()

        train_idx = np.linspace(0, self.n_epoch, len(self.train_loss_lst))
        valid_idx = np.arange(self.n_epoch)
        
        plt.subplot(2,1,1)
        plt.ylabel("Loss")
        plt.yscale(yscale)
        plt.plot(train_idx, self.train_loss_lst, c="r", label="train")
        plt.plot(valid_idx, self.valid_loss_lst, c="b", label="validate")
        plt.legend()

        plt.subplot(2,1,2)
        plt.ylim(0, 1.1)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(train_idx, self.train_acc_lst, c="r", label="train")
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
    
    def view_node_info(self):
        print([x.n_out for x in self.layerlst])
        
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
    
        
