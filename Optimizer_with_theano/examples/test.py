import Optimizer as op

o = op.optimizer()
o.set_datasets("mnist", is_one_hot=False)
o = o.sigmoid().dense(10)

o = o.loss_softmax_cross_entropy().opt_Adam(0.00001).optimize(100000, 100)

#o = op.optimizer(n_batch=40)
#o.set_datasets("mnist", is_one_hot=False)
##o = o.reshape((n_batch,1,28,28)).conv2d(kshape=(4,1,24,24)).relu().pool()
#o = o.reshape((1, 28, 28)) \
#    .conv_and_pool( 8,  5,  5)\
#    .conv_and_pool( 16,  5,  5)
#o = o.flatten().dropout(0.5).dense(10).loss_softmax_cross_entropy()
#o = o.opt_Adam(0.00000001).optimize(1000, 1)