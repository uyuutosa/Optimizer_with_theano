import Optimizer_with_theano as op
import numpy as np
import inspect



class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            vgg19_npy_path = "{}/param_dir/vgg19.npy".format("/".join(inspect.getfile(op.optimizer).split("/")[:-1]))
        self.data_dict = np.load(vgg19_npy_path, encoding="latin1").item()
    def build(self, obj, is_train=False):
        # Block 1
        obj = obj.conv2d(kshape=(64,3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv1_1"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv1_1"][1], 
                         is_train=is_train,
                         name="conv1_1")
        obj = obj.conv2d(kshape=(64,3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv1_2"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv1_2"][1], 
                         is_train=is_train,
                         name="conv1_2")
        obj = obj.pool(name="block1_pool")
        
        # Block 2
        obj = obj.conv2d(kshape=(128,3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv2_1"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv2_1"][1], 
                         is_train=is_train,
                         name="conv2_1")
        obj = obj.conv2d(kshape=(128,3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv2_2"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv2_2"][1], 
                         is_train=is_train,
                         name="conv2_2")
        obj = obj.pool(name="block2_pool")
        
        # Block 3
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv3_1"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv3_1"][1], 
                         is_train=is_train,
                         name="conv3_1")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv3_2"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv3_2"][1], 
                         is_train=is_train,
                         name="conv3_2")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv3_3"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv3_3"][1], 
                         is_train=is_train,
                         name="conv3_3")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv3_4"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv3_4"][1], 
                         is_train=is_train,
                         name="conv3_4")
        obj = obj.pool(name="block3_pool")
            
        # Block 4
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv4_1"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv4_1"][1], 
                         is_train=is_train,
                         name="conv4_1")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv4_2"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv4_2"][1], 
                         is_train=is_train,
                         name="conv4_2")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv4_3"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv4_3"][1], 
                         is_train=is_train,
                         name="conv4_3")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv4_4"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv4_4"][1], 
                         is_train=is_train,
                         name="conv4_4")
        obj = obj.pool(name="block4_pool")
        
        # Block 5
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv5_1"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv5_1"][1], 
                         is_train=is_train,
                         name="conv5_1")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv5_2"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv5_2"][1], 
                         is_train=is_train,
                         name="conv5_2")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv5_3"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv5_3"][1], 
                         is_train=is_train,
                         name="conv5_3")
        obj = obj.conv2d(kshape=(512, 3, 3), 
                         mode="same", 
                         act="relu", 
                         theta=self.data_dict["conv5_4"][0].transpose(3,2,0,1), 
                         b=self.data_dict["conv5_4"][1], 
                         is_train=is_train,
                         name="conv5_4")
        obj = obj.pool(name="block5_pool")
        return obj
        
            