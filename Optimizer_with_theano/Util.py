import theano 
import theano.tensor as T
import matplotlib.pyplot as plt
from .Datasets import gen_time_series
import numpy as np

def conv_ndarr_if_shared(arr):
    return arr if type(arr) is np.ndarray else arr.eval()

def conv_shared_if_ndarr(arr, dtype=theano.config.floatX):
    return theano.shared(arr.astype(dtype)) if type(arr) is np.ndarray else arr


def train_test_split(x_arr, y_arr, test_size, is_shuffle=None):
    is_x_shared = True if type(x_arr) is np.ndarray else False
    
    n_row = conv_ndarr_if_shared(x_arr).shape[0]
        
    if is_shuffle:
        idx = np.random.permutation(n_row)
        x_arr = x_arr[idx]
        y_arr = y_arr[idx]
        
    lim = int(n_row * (1 - test_size))
    x_train_arr = x_arr[:lim]
    x_test_arr  = x_arr[lim:]
    y_train_arr = y_arr[:lim]
    y_test_arr  = y_arr[lim:]
    train_n_row = lim
    test_n_row = n_row - lim
    
    return x_train_arr,\
           x_test_arr,\
           y_train_arr,\
           y_test_arr,\
           n_row,\
           train_n_row,\
           test_n_row
    
class Time_series_evaluator:
    def __init__(self, obj):
        self.obj = obj
    
    def series_predict(self, n_iter, initial_value=None):
        if initial_value is None:
            initial_value = self.obj.x_train_arr[0]
        lst = []
        for i in range(n_iter):
            v = self.obj.pred_func(np.array([initial_value])).flatten()
            initial_value = np.append(initial_value[1:],v)
            lst += [v[0]]
        plt.ylim(-1,1)
        plt.plot(lst)
        plt.show()
        
    def evaluate(self, data_length, predict_length):
        plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        is_first = True
        
        for i in range(1,10,1):
            plt.subplot(3, 3, i)
            v = np.zeros(1000)
            for i in range(10):
                v += np.sin(np.pi * np.arange(1000) / np.random.randint(10, 1000) + np.random.randint(0, 1000)) * np.arange(1000)[::-1]
            x, y, xidx, yidx = gen_time_series(v, data_length, predict_length)
            plt.plot(np.arange(v.size),v, c="r",lw=5, label="Grand truth")
            plt.plot(yidx, self.obj.pred_func(x.astype(theano.config.floatX)).flatten(), c="b", label="Predict")
            plt.plot(yidx, abs(y.flatten() - self.obj.pred_func(x.astype(theano.config.floatX)).flatten()), c="g", label="Error")
            plt.xlabel("x")
            plt.ylabel("y")
            if is_first:
                plt.legend()
                is_first = False
        plt.show()
        
class Image_classification_evaluator:
    def __init__(self, obj):
        self.obj = obj
        
    def evaluate(self,
                 data=None, 
                 label=None,
                 subplot=(3, 3, 1), 
                 img_shape=(28,28),
                 is_one_hot=True,
                 is_shuffle=True
                ):
        plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        
        if data is None:
            data  = self.obj.x_train_arr
            
        if label is None:
            label = self.obj.y_train_arr
            if is_one_hot:
                label = label.argmax(axis=1)[:,None]
                
        if is_shuffle:
            idx = np.random.permutation(data.shape[0])
            data = data[idx]
            label = label[idx]
        
        
        subplot = list(subplot)
        n_data = np.array(subplot).prod() + 1
        for i in range(1, n_data, 1):
            subplot[-1] = i
            plt.subplot(*subplot)

            img     = data[i-1]
            gt      = label[i-1][0] 
            predict = self.obj.pred_func([img])[0,0]
            plt.title("gt: {}, pred: {}, {}".format(gt, predict, gt==predict))
            plt.imshow(img.reshape(img_shape))
            plt.axis("off")
            
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.6, wspace=None, hspace=None)
        plt.show()