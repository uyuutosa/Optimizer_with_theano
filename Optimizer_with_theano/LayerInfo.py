import numpy as np

class Layer_info:
    """Manage the layers information.
    
    
    """
    
    def __init__(self):
        """Initialization of layers information.
        """
        
        self.layerlst = []
        self.layer_dic = {}
        self.obj_dic = {}
        self.dropout_rate_lst = []
        
        self.out_dic = {} 
        self.layer_num = 0
        
    def set_layer(self, layer):
        """Set a layer object.
        
        :param layer: the layer object
        """
        
        self.layerlst += [layer]
        self.layer_dic.update({layer.name:layer})
        self.layer_num += 1
        self.out_dic.update({layer.name:layer.obj.out})
        self.obj_dic.update({layer.name:layer.obj})
        
    def set_layers(self, layerlst):
        """Set a layer object.
        
        :param layer: the layer object
        """
        
        self.layerlst = [self.layerlst[0]]  + layerlst
        self.layer_num = len(self.layerlst)
        
    def update_layers(self, obj):
        for layer in self.layerlst:
            obj = layer.update(obj)
        return obj
    
    def set_dropout_param(self, rate, shared_rate):
        """Set a parameter of dropout.
        
        :param rate: the dropping rate(0 - 1)
        :param rate: the shared variables of the dropping rate(0 - 1)
        """
        self.dropout_rate_lst += [(rate, shared_rate)]
    
    def load_params(self, path, encoding="latin1"):
        param_dic = np.load(path, encoding)
        for k,v in param_dic.item():
            params = self.layer_dic[k].params
            for key, value in v:
                params[key].set_value(value)
            
        
    def get_shape_of_last_node(self):
        return self.layerlst[-1].n_out
    
    def get_params(self):
        lst = []
        for layer in self.layerlst:
            if len(layer.params):
                lst += list(layer.params.values())
        return lst
    
    def get_out(self, name):
        return self.out_dic[name]
    
    def get_layer(self, name):
        return self.layer_dic[name]
    
    def get_obj(self, name):
        return self.obj_dic[name]
        
    def view_info(self):
        """View the layers infomation.
        
        :param rate: the dropping rate(0 - 1)
        :param rate: the shared variables of the dropping rate
        """
        
        n_params = 0 
        line = "===================================="
        subline = "------------------------------------"
        for num, layer in enumerate(self.layerlst):
            print(line)
            print("layer {}".format(num))
            print("name:{}".format(layer.name))
            print("in:{}".format(layer.n_in))
            print("out:{}".format(layer.n_out))
            if len(layer.params):
                n_param = np.array([x.size.eval() for x in layer.params.values()]).sum()
                print("Num of params:{}".format(n_param))
                print("Params:")
                print(subline)
                for k, v in layer.params.items():
                    print("{}:{}".format(k,v.shape.eval()))
                print(subline)
                print("activation:{}".format(layer.actname))
                n_params += n_param
        print(line)
        print("Total num. of params:{}".format(n_params))
        print(line)
        
            