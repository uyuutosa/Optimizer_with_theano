# Optimizer_with_theano
Simple deep learning or machine learning framework for now.

Manual page is [here](https://uyuutosa.github.io/Optimizer_with_theano).

You can install as type

```sh
pip install Optimizer_with_theano
```

Test MNIST with multiclass logistic regression

```python
o = op.optimizer(128)
o = o.set_datasets()
o = o.dense(400).relu()
o = o.dense(30).relu()
o = o.dense(20).relu()
o = o.dense(10)
o = o.softmax().loss_cross_entropy()
o = o.opt_Adam(0.001).compile()
o = o.optimize(100, 10)
o.view()
```
```
Epoch. 0: loss = 8.2190e-01, acc = 7.2837e-01, valid. loss = 3.9006e-01, valid. acc. = 8.8486e-01.
Epoch. 10: loss = 6.5173e-02, acc = 9.8186e-01, valid. loss = 1.2214e-01, valid. acc. = 9.6129e-01.
Epoch. 20: loss = 2.7904e-02, acc = 9.9276e-01, valid. loss = 9.4638e-02, valid. acc. = 9.7014e-01.
Epoch. 30: loss = 1.1290e-02, acc = 9.9784e-01, valid. loss = 9.5740e-02, valid. acc. = 9.7343e-01.
Epoch. 40: loss = 4.4421e-03, acc = 9.9962e-01, valid. loss = 9.2676e-02, valid. acc. = 9.7586e-01.
Epoch. 50: loss = 1.6348e-03, acc = 9.9998e-01, valid. loss = 1.0095e-01, valid. acc. = 9.7643e-01.
Epoch. 60: loss = 6.9370e-04, acc = 1.0000e+00, valid. loss = 1.0945e-01, valid. acc. = 9.7629e-01.
Epoch. 70: loss = 3.0688e-04, acc = 1.0000e+00, valid. loss = 1.1512e-01, valid. acc. = 9.7657e-01.
Epoch. 80: loss = 1.4491e-04, acc = 1.0000e+00, valid. loss = 1.2495e-01, valid. acc. = 9.7700e-01.
Epoch. 90: loss = 6.0175e-05, acc = 1.0000e+00, valid. loss = 1.3110e-01, valid. acc. = 9.7743e-01.
```

or use CNN (a little)

```python
#One linear coding suppotted(but I do not use it)..
import Optimizer_with_theano as op
o = op.optimizer(128)\
      .set_datasets()\
      .reshape((1,28,28))\
      .conv_and_pool(64,3,3, "same")\
      .conv_and_pool(32,3,3, "same")\
      .flatten().dense(10)\
      .softmax().loss_cross_entropy()\
      .opt_Adam(0.001).compile()\
      .optimize(100, 1)
```

```
Epoch. 0: loss = 6.8408e-01, acc = 7.8419e-01, valid. loss = 2.1394e-01, valid. acc. = 9.3871e-01.
Epoch. 10: loss = 4.2716e-02, acc = 9.8716e-01, valid. loss = 5.1318e-02, valid. acc. = 9.8314e-01.
Epoch. 20: loss = 2.9371e-02, acc = 9.9140e-01, valid. loss = 3.8624e-02, valid. acc. = 9.8800e-01.
Epoch. 30: loss = 2.0271e-02, acc = 9.9408e-01, valid. loss = 4.1340e-02, valid. acc. = 9.8671e-01.
Epoch. 40: loss = 1.4117e-02, acc = 9.9594e-01, valid. loss = 3.9778e-02, valid. acc. = 9.8729e-01.
Epoch. 50: loss = 9.5171e-03, acc = 9.9760e-01, valid. loss = 4.4686e-02, valid. acc. = 9.8729e-01.
Epoch. 60: loss = 7.2461e-03, acc = 9.9819e-01, valid. loss = 5.0205e-02, valid. acc. = 9.8729e-01.
Epoch. 80: loss = 2.3437e-03, acc = 9.9971e-01, valid. loss = 5.7557e-02, valid. acc. = 9.8614e-01.
Epoch. 90: loss = 1.5257e-03, acc = 9.9987e-01, valid. loss = 6.4261e-02, valid. acc. = 9.8557e-01.
```

We try to simple time series prediction using blow data,

```python
from numpy import *
from pylab import *
import Optimizer as op
%matplotlib inline

def gen_dataset(v, length):
    v = v.flatten()
    idx = arange(v.size)
    idx = idx[:, None] + arange(length+1)
    idx = idx[:-length]
    xidx = idx[:, :-1]
    yidx = idx[:, -1]
    x_idx_f = xidx.flatten()
    y_idx_f = yidx.flatten()
    x = v[x_idx_f].reshape(-1, 1, 1, length)
    y = v[y_idx_f][:, None]
    return x, y, xidx, yidx

v = sin(pi * arange(10000) / arange(1,10001)[::-1]*10) * arange(10000)[::-1]
x, y, xidx, yidx = gen_dataset(v, 10)

plot(arange(v.size),v)
plot(yidx.flatten(),y.flatten())
xlabel("x")
ylabel("y")
legend(["data", "labels"])
```

![spring](images/spring.png)

Trying to learn this data using bellow network(so called UFCNN)

```python
o = op.optimizer(n_batch=100)
o.set_data(x, y, test_size=0., is_shuffle=False)
o.set_variables()
o1 =  o.conv2d((1, 1, 1, 5), mode="same").relu()
o2 = o1.conv2d((1, 1, 1, 5), mode="same").relu()
o3 = o2.conv2d((1, 1, 1, 5), mode="same").relu()
o4 = o3.conv2d((1, 1, 1, 5), mode="same").relu()
o5 = o4.conv2d((1, 1, 1, 5), mode="same").relu() + o2
o6 = o5.conv2d((1, 1, 1, 5), mode="same").relu() + o1
o7 = o6.conv2d((1, 1, 1, 5), mode="same").relu()
o8 = o7.conv2d((1, 1, 1, 10), mode="valid")
o9 = o8.flatten()
o = o9.loss_mse()
o = o.opt_Adam(0.001).compile()
o = o.optimize(10000000,10)
```

Loss value can be viewed by



```python
o.view()
```

![drastic](images/drop.png)

and comparing between predict and grand truth using dataset for checking.

```python
#plot(xidx.flatten(),x.flatten())
plot(arange(v.size),v)
#plot(yidx.flatten(),y.flatten())
plot(yidx, o.pred_func(x.reshape(-1, 1, 1, 10).astype(float32)).flatten(), c="b")
#xlim(0,100)
xlabel("x")
ylabel("y")
legend(["data", "predict"])
```


![compareing](images/true_pred.png)


Lastly, evaluate the performance of this prediction model using randomly generated data as follows.

```python
figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
is_first = True
for i in range(1,10,1):
    subplot(3, 3, i)
    v = zeros(1000)
    for i in range(10):
        v += sin(pi * arange(1000) / randint(10, 1000) + randint(0, 1000)) * arange(1000)[::-1]
    x, y, xidx, yidx = gen_dataset(v, 10)
    plot(arange(v.size),v, c="r",lw=5, label="Grand truth")
    plot(yidx, o.pred_func(x.reshape(-1, 1, 1, 10).astype(float32)).flatten(), c="b", label="Predict")
    plot(yidx, abs(y.flatten() - o.pred_func(x.reshape(-1, 1, 1, 10).astype(float32)).flatten()), c="g", label="Error")

    xlabel("x")
    ylabel("y")
    if is_first:
        legend()
        is_first = False

```

![true_pred_err](images/true_pred_err.png)

This model seemed to be able to predit exactly.
However, it should be noted that this coarse evaluation does not serve as any reference.
This is a toy problem.
