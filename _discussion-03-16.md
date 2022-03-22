+ Paddle 2.0版本中封装了新的高层API，多数API调用方式及参数名参考了PyTorch和TensorFlow。参考：[飞桨高层API使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/quick_start/high_level_api/high_level_api.html)

+ paddle.fluid是Paddle 1.x的产物，文档中提示即将废弃。

  

+ **paddle.nn.ReLU() 和 paddle.nn.functional.relu()**

  与PyTorch中一样，paddle.nn和torch.nn包含了组网时可能被使用到的各种components，Paddle中的paddle.nn.ReLu()继承了paddle.nn.Layer类，调用时作为类被调用（PyTorch中的torch.nn.ReLU()继承了torch.nn.Module类，Paddle的设计思路显然很大程度上借鉴了PyTorch）。
  
  在Paddle和PyTorch中，nn.functional.relu()被作为函数定义，调用时作为一个无参函数被调用，更加简洁。
  
  可以从Paddle的源码中观察到这两者的具体实现（[paddle.nn.functional.relu()](https://github.com/PaddlePaddle/Paddle/blob/release/2.2/python/paddle/nn/functional/activation.py#L492)与[paddle.nn.ReLU()](https://github.com/PaddlePaddle/Paddle/blob/release/2.2/python/paddle/nn/layer/activation.py#L399)）：
  
  ```python
  # paddle.nn.functional.relu()
  def relu(x, name=None):
      
      if in_dygraph_mode():
          return _C_ops.relu(x)
      
      check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'relu')
      helper = LayerHelper('relu', **locals())
      out = helper.create_variable_for_type_inference(x.dtype)
      helper.append_op(type='relu', inputs={'X': x}, outputs={'Out': out})
      return out
  ```
  
  ```python
  # paddle.nn.ReLU()
  class ReLU(Layer):
      
      def __init__(self, name=None):
          super(ReLU, self).__init__()
          self._name = name
          
      def forward(self, x):
          # from .. import functional as F
          return F.relu(x, self._name)
      
      def extra_repr(self):
          name_str = 'name={}'.format(self._name) if self._name else ''
          return name_str
  ```
  
  可以看到paddle.nn下的ReLU类在forward时直接调用了nn.functional下的relu()函数，而最终的计算则是relu()函数通过调用C++组件库中的相应部分进行计算的。
  
  两句简单的示例：
  
  ```python
  from paddle import nn
  from paddle.nn import functional as F
  
  X = paddle.to_tensor([-1.0, 0.0, 1.0])
  ```
  
  ```python
  F.relu(X) # will return [0., 0., 1.]
  ```
  
  ```python
  m = paddle.nn.ReLU()
  m(X) # will return [0., 0., 1.], same as before
  ```
  
+ **paddle.nn 与 paddle.nn.functional**

  paddle.nn中大部分组件在paddle.nn.functional中都有一个对应的函数，如nn.Conv2D()和nn.functional.conv2d()。Paddle这方面与PyTorch设计一致，通过同时保留这两者保证了使用时的灵活性。 

  参考：[PyTorch 中，nn 与 nn.functional 有什么区别？](https://www.zhihu.com/question/66782101)

  另外，使用这两者时模型的结构有一些细微的差别，可以通过一个简单的例子观察到：

```python
import paddle
from paddle import nn
from paddle.nn import functional as F

class TestNet(nn.Layer):
    def __init__(self, n_feature, n_hidden, n_output):
        super(TestNet, self).__init__()
        self.linear1 = nn.Linear(n_feature, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

net1 = TestNet(1, 10, 1)

net2 = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
```

```python
model1 = paddle.Model(net1)
model1.summary((10, 1))
"""
---------------------------------------------------------------------------
 Layer (type)       Input Shape          Output Shape         Param #    
===========================================================================
   Linear-3          [[10, 1]]             [10, 10]             20       
   Linear-4          [[10, 10]]            [10, 1]              11       
===========================================================================
Total params: 31
Trainable params: 31
Non-trainable params: 0
---------------------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
---------------------------------------------------------------------------

{'total_params': 31, 'trainable_params': 31}
"""
```

```python
model2 = paddle.Model(net2)
model2.summary((10, 1))
"""
---------------------------------------------------------------------------
 Layer (type)       Input Shape          Output Shape         Param #    
===========================================================================
   Linear-5          [[10, 1]]             [10, 10]             20       
    ReLU-7           [[10, 10]]            [10, 10]              0       
   Linear-6          [[10, 10]]            [10, 1]              11       
===========================================================================
Total params: 31
Trainable params: 31
Non-trainable params: 0
---------------------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
---------------------------------------------------------------------------

{'total_params': 31, 'trainable_params': 31}
"""
```

  可以观察到：对于在forward中使用了nn.functional.relu()函数的net1，打印出的网络结构中只有两个线性层；而对于在模型结构定义中将nn.ReLU()定义为自身属性的net2，打印出的网络结构中则将ReLU视为网络中的一个无参的层。

+ 显然，在用Sequential容器定义网络模块时，必须传入nn.Layer类型的sublayer或可迭代的nn.Layer元组。
+ **[Paddle中模型训练的若干方法](https://blog.csdn.net/lxm12914045/article/details/111505366)**

  以LeNet在Fashion-MNIST上为例：

```python
import paddle
import numpy as np
from paddle import nn
from paddle.nn import functional as F

import paddle.vision.transforms as T
from paddle.vision.datasets import FashionMNIST

# loading and normalization
transform = T.Normalize(mean=[127.5], std=[127.5])  

# constructing traning set and test set
fashionmnist_train = FashionMNIST(mode='train', transform=transform)
fashionmnist_test = FashionMNIST(mode='test', transform=transform)

train_loader = paddle.io.DataLoader(fashionmnist_train, batch_size=256, shuffle=True)
test_loader = paddle.io.DataLoader(fashionmnist_test, batch_size=64, shuffle=False)

net = nn.Sequential(
    nn.Conv2D(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2D(kernel_size=2, stride=2),
    nn.Conv2D(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2D(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

model = paddle.Model(net)

# optimizer, loss and metrics
model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-4),
              loss=nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())
```

```python
# 1
# type(fashionmnist_train): Dataset
model.fit(fashionmnist_train,
          epochs=3,
          batch_size=64,
          verbose=1)

model.evaluate(fashionmnist_test, verbose=1)
```

```python
# 2
# type(train_loader): Dataloader
model.fit(train_loader,
        # eval_data = test_loader
        epochs=3,
        verbose=1,
        )

model.evaluate(fashionmnist_test, verbose=1)
```

```python
# 3
for epoch in range(2):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = data[1]
        info = model.train_batch([x_data], [y_data])

        if batch_id % 10 == 0:
            print(info)

    for ebatch_id, edata in enumerate(test_loader()):
        x_data_v = edata[0]
        y_data_v = edata[1]
        info = model.eval_batch([x_data_v], [y_data_v])

        if ebatch_id % 10 == 0:
            print(info)

model.evaluate(fashionmnist_test, verbose=1)

```

```python
# 4
net.train()
optim = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())

for epoch in range(2):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = data[1]
       
        predicts = net(x_data)
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data, k=2)
        loss.backward()
        if batch_id % 10 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
        optim.step()
        optim.clear_grad()

    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        
        predicts = net(x_data)
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data, k=2)
        if batch_id % 10 == 0:
            acc = paddle.metric.accuracy(predicts, y_data, k=2)
        if batch_id % 10 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
```

```python
# 4

# training
net.train()
epochs = 5
optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=1e-3)
for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = data[1]

        predicts = net(x_data)
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        loss.backward()
        if batch_id % 200 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
        optimizer.step()
        optimizer.clear_grad()

# testing
net.eval()
for batch_id, data in enumerate(test_loader()):
    x_data = data[0]
    y_data = data[1]
    predicts = net(x_data)

    loss = F.cross_entropy(predicts, y_data)
    acc = paddle.metric.accuracy(predicts, y_data)
    if batch_id % 50 == 0:
        print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))
```

```python
# 5

val_acc_history = []
val_loss_history = []

def train(model):
    print('start training ... ')
    # training mode on
    model.train()

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-3)
    train_loader = paddle.io.DataLoader(fashionmnist_train, batch_size=128, shuffle=True)
    test_loader = paddle.io.DataLoader(fashionmnist_test, batch_size=128, shuffle=False)
    epochs = 3

    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]

            predicts = net(x_data)
            loss = F.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 200 == 0:
                print("[train] epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optimizer.step()
            optimizer.clear_grad()

        # evaluate model after every epoch
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(test_loader()):
            x_data = data[0]
            y_data = data[1]

            predicts = net(x_data)
            loss = F.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[validation] loss is: {}, acc is: {}".format(avg_loss, avg_acc))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        model.train()

train(net)
```

```python
start training ... 
[train] epoch: 0, batch_id: 0, loss is: [0.38968918], acc is: [0.875]
[train] epoch: 0, batch_id: 200, loss is: [0.4175303], acc is: [0.8828125]
[train] epoch: 0, batch_id: 400, loss is: [0.48383802], acc is: [0.828125]
[validation] loss is: 0.3646795451641083, acc is: 0.8670886158943176
[train] epoch: 1, batch_id: 0, loss is: [0.2479319], acc is: [0.9140625]
[train] epoch: 1, batch_id: 200, loss is: [0.19505194], acc is: [0.9453125]
[train] epoch: 1, batch_id: 400, loss is: [0.33912522], acc is: [0.8984375]
[validation] loss is: 0.3558771014213562, acc is: 0.8701542615890503
[train] epoch: 2, batch_id: 0, loss is: [0.416911], acc is: [0.8203125]
[train] epoch: 2, batch_id: 200, loss is: [0.277215], acc is: [0.9140625]
[train] epoch: 2, batch_id: 400, loss is: [0.23891735], acc is: [0.90625]
[validation] loss is: 0.35426172614097595, acc is: 0.8693631291389465
```

