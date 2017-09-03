from sklearn.datasets import *
from sklearn.datasets import fetch_mldata
import numpy as np

def gen_one_hot(idx):
    num = int(idx.max()+1)
    arr = np.zeros((idx.shape[0],num)).flatten()
    arr[idx.flatten().astype(int) + np.arange(idx.shape[0]) * num]  = 1
    return arr.reshape(idx.size, num)

def gen_time_series(v, data_length, predict_length):
    v = v.flatten().astype("float32")
    idx = np.arange(v.size)
    idx = idx[:, None] + np.arange(data_length+predict_length)
    idx = idx[:-data_length]
    xidx = idx[:, :-predict_length]
    yidx = idx[:, -predict_length:]
    x_idx_f = xidx[:-predict_length].flatten()
    y_idx_f = yidx[:-predict_length].flatten()
    x = v[x_idx_f].reshape(-1, data_length)
    #x = v[x_idx_f].reshape(-1, 1, 1, length)
    y = v[y_idx_f].reshape(-1, predict_length)
    return x, y, xidx[:-predict_length], yidx[:-predict_length]
 
def set_datasets(data="mnist", 
                 is_one_hot=True, 
                 is_normalize=True, 
                 **kwarg):
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
        if is_normalize == True:
            data_dic["data"] = data_dic["data"] / data_dic["data"].max(axis=0)
    elif data == "digits":
        data_dic = load_digits()
    elif data == "iris":
        data_dic = load_iris()
        if is_one_hot == True:
            data_dic["target"] = gen_one_hot(data_dic["target"])
        if is_normalize == True:
            data_dic["data"] = data_dic["data"] / data_dic["data"].max(axis=0)
    elif data == "linnerud":
        data_dic = load_linnerud()
    elif data == "wine":
        arr = np.loadtxt(data_home + "/wine.csv", delimiter=",", skiprows=1)
        data_dic = {"data": arr[:, :-1], "target":arr[:, -1]}
        if is_one_hot == True:
            data_dic["target"] = gen_one_hot(data_dic["target"])
        if is_normalize == True:
            data_dic["data"] = data_dic["data"] / data_dic["data"].max(axis=0)
    elif data == "xor":
        data_dic = {"data": np.array([[0,0], [0,1], [1,0], [1,1]]),##.repeat(20, axis=0),
                        "target": np.array([0,1,1,0])}#.repeat(20, axis=0)}
    elif data == "serial":
        data_dic = {"data": np.array(np.arange(20).reshape(5,4)).repeat(20, axis=0),
                        "target": np.arange(5).repeat(20, axis=0)}
   # elif data == "sin":
   #     data_dic = {"data": np.arange(0,10,0.01)[:,None],
   #                     "target": np.sin(np.arange(0,10,0.01) * np.pi)}
   # 
   # elif data == "sin":
   #     
   #     data_dic = {"data": np.arange(0,10,0.01)[:,None],
   #                     "target": np.sin(np.arange(0,10,0.01) * np.pi)}
    elif data == "sin":
        v = np.sin(np.pi * np.arange(1000) / 100) 
        if not "data_length" in kwarg:
            data_length = 100
        else:
            data_length = kwarg["data_length"]
        
        if not "predict_length" in kwarg:
            predict_length = 1
        else:
            data_length = kwarg["data_length"]
          
        x, y, xidx, yidx = gen_time_series(v, data_length, predict_length)
        data_dic = {"data": x, "target": y}
    elif data == "decay":
        v = np.sin(np.pi * np.arange(10000) / np.arange(1,10001)[::-1]*10) * np.arange(10000)[::-1]
        v = v[:-1000]
        x, y, xidx, yidx = gen_time_series(v, 10, 1)
        data_dic = {"data": x, "target": y}
        
    if "data_only"in kwarg:
        data_dic["target"] = data_dic["data"]
        
    return data_dic["data"], data_dic["target"]
