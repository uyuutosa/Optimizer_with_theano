import theano
import theano.tensor as T
import numpy as np
from .Util import conv_shared_if_ndarr

class Compile_and_optimize:
    def __init__(self, obj, is_random=False):
        self.obj = obj
        self.is_random = is_random
        
    def compile(self):
        pass
    
    def optimize(self):
        pass
        
class CO_fast_but_takes_a_lot_of_memory(Compile_and_optimize):
    def __init__(self, obj, is_random=True):
        super().__init__(obj, is_random)
        
    def compile(self):
        obj = self.obj
        obj.dsize = obj.train_n_row

        i = theano.shared(0).astype("int32")
        
        if self.is_random:
            obj.idx = theano.shared(np.random.permutation(obj.train_n_row))
        else:
            obj.idx = theano.shared(np.arange(obj.train_n_row))
            
        train_xgivens = []
        for t, train_xgiven in zip(obj.xlst, obj.train_xgivenlst):
            xgiven_shared = conv_shared_if_ndarr(train_xgiven)
            train_xgivens += [(t, xgiven_shared[obj.idx[i:obj.n_batch+i],])]
        train_ygivens = []
        for t, train_ygiven in zip(obj.ylst, obj.train_ygivenlst):
            ygiven_shared = conv_shared_if_ndarr(train_ygiven)
            train_ygivens += [(t, ygiven_shared[obj.idx[i:obj.n_batch+i],])]
            
        test_xgivens = []
        for t, test_xgiven in zip(obj.xlst, obj.test_xgivenlst):
            xgiven_shared = conv_shared_if_ndarr(test_xgiven)
            test_xgivens += [(t, xgiven_shared)]
            
        test_ygivens = []
        for t, test_ygiven in zip(obj.ylst, obj.test_ygivenlst):
            ygiven_shared = conv_shared_if_ndarr(test_ygiven)
            test_ygivens += [(t, ygiven_shared)]
            
        obj.train_loss_and_acc = theano.function(inputs=[i],
                                        outputs=[obj.loss, obj.train_acc],
                                        givens=train_xgivens+train_ygivens,
                                        updates=obj.updatelst,
                                        on_unused_input='ignore')
        
        obj.valid_loss_and_acc = theano.function(inputs=[],
                                        outputs=[obj.loss, obj.valid_acc],
                                        givens=test_xgivens+test_ygivens,
                                        on_unused_input='ignore')
        
        
        obj.pred_func = theano.function(inputs  = obj.xlst,#[obj.x],
                                        outputs = obj.out,
                                        updates = obj.tmplst,
                                        on_unused_input='ignore')
    
        return obj

    def optimize(self, 
                 n_epoch=100, 
                 n_view=10, 
                 n_iter=None, 
                 n_batch=None, 
                 is_valid=True,
                 is_view=True):
        
            
        obj = self.obj
        
        if n_batch is not None:
            obj.n_batch.set_value(n_batch)
            
        obj.layer_info.view_info()
        
        if obj.n_view is None: 
            obj.n_view = n_view  
            
        obj.train_loss_lst = [] 
        obj.train_acc_lst  = []
        obj.valid_loss_lst = []
        obj.valid_acc_lst  = []
        try:
            obj.n_epoch = n_epoch
            for epoch in range(n_epoch):
                if self.is_random:
                    tmp = np.random.permutation(obj.train_n_row)
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
                    
                
                if is_valid:
                    [x[1].set_value(0) for x in obj.dropout_rate_lst]
                    valid_loss_value, valid_acc_value = obj.valid_loss_and_acc()
                    [x[1].set_value(x[0]) for x in obj.dropout_rate_lst]
                    obj.valid_loss_lst   += [valid_loss_value]
                    obj.valid_acc_lst    += [valid_acc_value]
                train_mean_loss_value = np.array(train_loss_lst).mean()
                train_mean_acc_value  = np.array(train_acc_lst).mean()
                obj.train_loss_lst   += train_loss_lst
                obj.train_acc_lst    += train_acc_lst
                if not (epoch % n_view):
                    if is_valid:
                        print("Epoch. %s: loss = %.4e, acc = %.4e, valid. loss = %.4e, valid. acc. = %.4e." %(epoch, 
                                                                                                        train_mean_loss_value, 
                                                                                                        train_mean_acc_value, 
                                                                                                        valid_loss_value,
                                                                                                        valid_acc_value))
                    else:
                        print("Epoch. %s: loss = %.4e, acc = %.4e." %(epoch,
                                                                      train_mean_loss_value, 
                                                                      train_mean_acc_value, 
                                                                      ))
            
            
        except KeyboardInterrupt:
            print ( "KeyboardInterrupt\n" )
            obj.n_epoch = epoch
            if is_view:
                obj.view(is_valid=is_valid)
            return obj
        
        if is_view:
            obj.view(is_valid=is_valid)
            
        return obj
    
class CO_slow_but_only_few_memory_needed(Compile_and_optimize):
    def __init__(self, obj, is_random=True):
        super().__init__(obj, is_random)
    
    def compile(self):
        obj = self.obj
        obj.dsize = obj.train_n_row

        i = theano.shared(0).astype("int32")
        
        if self.is_random:
            obj.idx = np.random.permutation(obj.train_n_row)
        else:
            obj.idx = np.arange(obj.train_n_row)
            
        input_lst = obj.xlst + obj.ylst
        obj.train_loss_and_acc = theano.function(inputs=input_lst,
                                                 outputs=[obj.loss, obj.train_acc],
                                                 updates=obj.updatelst,
                                                 on_unused_input='ignore')
        
        obj.valid_loss_and_acc = theano.function(inputs=input_lst,
                                                 outputs=[obj.loss, obj.valid_acc],
                                                 on_unused_input='ignore')
        
        obj.pred_func = theano.function(inputs  = obj.xlst,
                                        outputs = obj.out,
                                        updates = obj.tmplst,
                                        on_unused_input='ignore')
        return obj
        
    def optimize(self, 
                 n_epoch=100, 
                 n_view=10, 
                 n_iter=None, 
                 n_batch=None, 
                 is_valid=True,
                 is_view=True):
        obj = self.obj
        obj.layer_info.view_info()
        
        if obj.n_view is None: 
            obj.n_view = n_view  
            
        obj.train_loss_lst = [] 
        obj.train_acc_lst  = []
        obj.valid_loss_lst = []
        obj.valid_acc_lst  = []
        
        if n_batch is None:
            n_batch = int(obj.n_batch.get_value())
            
        print("batch size:%s" %n_batch)
        try:
            obj.n_epoch = n_epoch
            for epoch in range(n_epoch):
                if self.is_random:
                    obj.idx = np.random.permutation(obj.train_n_row)
                
                mean_loss = 0.
                N = obj.dsize-n_batch + 1
                    
                train_loss_lst = []
                train_acc_lst = []
                for i in range(0, N, n_batch):
                    train_loss_value, train_acc_value =\
                        obj.train_loss_and_acc(*([x[obj.idx[i:i+n_batch]] for x in obj.train_xgivenlst] +\
                                               [y[obj.idx[i:i+n_batch]] for y in obj.train_ygivenlst])
                                              )
                    train_loss_lst += [train_loss_value]
                    train_acc_lst  += [train_acc_value]
                    
                
                if is_valid:
                    [x[1].set_value(0) for x in obj.dropout_rate_lst]
                    valid_loss_value, valid_acc_value = obj.valid_loss_and_acc(*(obj.test_xgivenlst+obj.test_ygivenlst))
                    [x[1].set_value(x[0]) for x in obj.dropout_rate_lst]
                    obj.valid_loss_lst   += [valid_loss_value]
                    obj.valid_acc_lst    += [valid_acc_value]
                
                train_mean_loss_value = np.array(train_loss_lst).mean()
                train_mean_acc_value  = np.array(train_acc_lst).mean()
                obj.train_loss_lst   += train_loss_lst
                obj.train_acc_lst    += train_acc_lst
                if not (epoch % n_view):
                    if is_valid:
                        print("Epoch. %s: loss = %.4e, acc = %.4e, valid. loss = %.4e, valid. acc. = %.4e." %(epoch, 
                                                                                                        train_mean_loss_value, 
                                                                                                        train_mean_acc_value, 
                                                                                                        valid_loss_value,
                                                                                                        valid_acc_value))
                    else:
                        print("Epoch. %s: loss = %.4e, acc = %.4e." %(epoch,
                                                                      train_mean_loss_value, 
                                                                      train_mean_acc_value, 
                                                                      ))
            
            
        except KeyboardInterrupt:
            print ( "KeyboardInterrupt\n" )
            obj.n_epoch = epoch
            if is_view:
                obj.view(is_valid=is_valid)
            return obj
        
        if is_view:
            obj.view(is_valid=is_valid)
            
        return obj
