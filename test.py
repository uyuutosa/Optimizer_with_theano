from PIL import Image
#from cupy import *
from numpy import *
import pylab as p
fname = "parrot.jpg" 
raw  = array(Image.open("parrot.jpg").convert("L"))
import cv2
import Optimizer_with_theano as op
img = cv2.resize(raw, (128, 128))
x = img / 255
x = x[None,None,...]
#x = x.repeat(10,axis=0)
o = op.optimizer(1)
o = o.set_data(x,x, test_size=0)
o = o.conv2d((64,5,5), mode="same", act="relu")
o = o.pool()
o = o.conv2d((128,5,5), mode="same", act="relu")
o = o.pool()
o = o.conv2d((256,5,5), mode="same", act="relu")
o = o.pool()
o = o.conv2d((512,5,5), mode="same", act="relu")
o = o.unpool()
o = o.conv2d((256,5,5), mode="same", act="relu")
o = o.unpool()
o = o.conv2d((128,5,5), mode="same", act="relu")
o = o.unpool()
o = o.conv2d((64,5,5), mode="same", act="relu")
o = o.conv2d((1,5,5), mode="same", act="sigmoid")
o = o.loss_binary_cross_entropy()
o = o.opt_Adam(0.0001).compile()
o = o.optimize(100,10,is_valid=False)
