from sklearn.datasets import *
from sklearn.datasets import fetch_mldata
import numpy as np

 
def set_datasets(data="mnist", is_one_hot=True, is_normalize=True):
    data_home="/".join(__file__.split("/")[:-1])+"/data_dir_for_optimizer"
    if data == "mnist":
        data_dic = fetch_mldata('MNIST original', data_home=data_home)
        if is_one_hot == True:
            idx = data_dic["target"]
            num = int(idx.max()+1)
            arr = np.zeros((idx.shape[0],num)).flatten()
            arr[idx.flatten().astype(int) + np.arange(idx.shape[0]) * num]  = 1
            data_dic["target"] = arr.reshape(idx.size, num)
        if is_normalize == True:
            data_dic["data"] = data_dic["data"] / 255
    elif data == "boston":
        data_dic = load_boston()   
    elif data == "digits":
        data_dic = load_digits()
    elif data == "iris":
        data_dic = load_iris()
    elif data == "linnerud":
        data_dic = load_linnerud()
    elif data == "xor":
        data_dic = {"data": np.array([[0,0], [0,1], [1,0], [1,1]]),##.repeat(20, axis=0),
                        "target": np.array([0,1,1,0])}#.repeat(20, axis=0)}
    elif data == "serial":
        data_dic = {"data": np.array(np.arange(20).reshape(5,4)).repeat(20, axis=0),
                        "target": np.arange(5).repeat(20, axis=0)}
    elif data == "sin":
        data_dic = {"data": np.arange(0,10,0.01)[:,None],
                        "target": np.sin(np.arange(0,10,0.01) * np.pi)}
    
        
    return data_dic["data"], data_dic["target"]
