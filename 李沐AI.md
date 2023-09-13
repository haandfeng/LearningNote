李沐AI

![截屏2023-03-15 11.31.53](/Users/ha/Library/Application Support/typora-user-images/截屏2023-03-15 11.31.53.png)

## 数据处理

```
x= torch.arange(12) #生成0-12的数组
x.shape
x.numel #数字的数量
x.reshape
x**y 是 x^y
torch.cat((x,y),dim=0or1) # 0 是横着合1时竖着合
x.sum 
```

运用插值算法，补充缺失的数据，详见04章第二节

## 线性代数

```
A.T #A是矩阵，矩阵的转置
A.clone() #复制
torch.mv(A,x) #两个矩阵相乘
torch.norm(A.reshape(1,width*len))#求模长 只能对向量，对矩阵要拉成向量再求
```

![截屏2023-03-15 19.09.08](/Users/ha/Library/Application Support/typora-user-images/截屏2023-03-15 19.09.08.png)

## 求导

### y是标量 x是向量

 这就是梯度，指向值变大的方向

![截屏2023-03-15 19.41.11](/Users/ha/Library/Application Support/typora-user-images/截屏2023-03-15 19.41.11.png)

### y是向量 x是标量

![截屏2023-03-15 19.44.14](/Users/ha/Library/Application Support/typora-user-images/截屏2023-03-15 19.44.14.png)

### 两个都是向量

![截屏2023-03-15 19.44.40](/Users/ha/Library/Application Support/typora-user-images/截屏2023-03-15 19.44.40.png)

计算基本定理，后面两个比较重要

![截屏2023-03-15 19.45.41](/Users/ha/Library/Application Support/typora-user-images/截屏2023-03-15 19.45.41.png)

## 求导的实现

哪怕是控制流，也可以实现梯度ß *** ？看视频***

```
# y关于向量x的导数
x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None 用来访问梯度
y = 2 * torch.dot(x, x)
y.backward()  # 反向传播就是求导
x.grad #tensor([ 0.,  4.,  8., 12.])  答案：x.grad == 4 * x
```

```
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum() # 求和的梯度
y.backward()
x.grad
```

```
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。？
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的？
x.grad.zero_()
y = x * x #y变成了向量
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

```
x.grad.zero_()
y = x * x
# 把y变成了一个常量，和x没关系
u = y.detach()
# 不再是x*x*x
z = u * x

z.sum().backward()
x.grad == u
```

## 线性回归

可以看成单层的神经网络

### 优化方法

#### 梯度下降

随机挑选初始值，重复迭代

学习率即在梯度方向上走了多少，不能太小也不能太大，太小会耗费时间，太大会震荡

#### 小批量随机梯度下降

随机选取b个样本来近似损失

太小不适合利用计算资源

#### 线性回归详细代码实现

```
import random
import torch
from d2l import dtorch as d2l


def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    # 均值为0，方差为1 的随机数
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    # 随机噪音
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    #  每个样本的index
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    # 打乱下标顺序 访问样本
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        # python 的iterate 每次返回一个一个x 一个y
        yield features[batch_indices], labels[batch_indices]


# 模型的定义
def linreg(X, w, b):  # @save
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数 即要优化的目标
def squared_loss(y_hat, y):  # @save
    """均方损失"""
    # 没求均值
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法 param 是所有的参数，即y=f(x)中的x, 这里是 w和b ，lr 是学习率
def sgd(params, lr, batch_size):  # @save
    """小批量随机梯度下降"""
    # 更新的时候不要有梯度计算
    with torch.no_grad():
        for param in params:
            #   batch_size 是求均值 上面的损失函数没有均值
            #   w和b就被更新了
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 真实的权重w和真实的b
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成训练样本
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])
d2l.set_figsize()
#  detach 才可以转换成numpy
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()
batch_size = 10
#
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 开始模型的训练
#  初始化回归模型，需要梯度保存
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.03
# 走多少遍
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    # 每一次拿出批量大小的x和y
    for X, y in data_iter(batch_size, features, labels):
        # 求损失
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        # 求和算梯度,让损失函数的和最小，l是关于（w，b）的损失函数，backward求出w和b的梯度，然后梯度下降将损失函数最小化
        l.sum().backward()
        # 对w和b进行更新
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
```

#### 线性回归简单代码实现

```
import numpy as np
import torch
from torch.utils import data
from d2l import dtorch as d2l
# nn是神经网络的缩写
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    # * 号表示解开参数，分开
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成数据
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# Sequential 即list of layer 这里只有一个layer
# nn.Linear(2, 1) 指定了输入的纬度和输出的纬度 linear表示一个单层的神经网络
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数 等价于前面的初始化

# 找到weight的data 用正态分布——normal 替换掉那里的值 即初始化
net[0].weight.data.normal_(0, 0.01)
# bias 即偏差
net[0].bias.data.fill_(0)

# 计算损失——均方误差
loss = nn.MSELoss()
# 进行优化
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 迭代计算
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        # 梯度清0
        trainer.zero_grad()
        #  已经做了sum
        l.backward()
        # 模型更新
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

```

## Softmax回归

#### 分类模型和回归模型的区别

![IMG_5B54D52C210A-1](/Users/ha/Downloads/IMG_5B54D52C210A-1.jpeg)

#### 实现方法 oi是置信度

![1651679131964_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1651679131964_.pic_hd.jpg)

#### 对计算得出来该结果的概率

![1671679132270_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1671679132270_.pic_hd.jpg)

#### 计算损失

y里面只有一个为1 其他为0， 所以可以化简

![1681679132338_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1681679132338_.pic_hd.jpg)

## 常用损失函数

常见评价标准就是看距离原点远近的时候的梯度

#### 均方损失 L2 Loss

- 蓝色 y=0 预测值y‘变化

- 绿色 似然函数？？ 高斯分布

- 橙色 损失函数的梯度 

  不断朝原点靠近，但距离远点远的时候梯度大 

  

![1701679132978_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1701679132978_.pic_hd.jpg)

#### L1 Loss

梯度永远是常数一个是1 一个是-1  但零点不可导 且存在突变，不平滑，优化到末期不稳定

![1721679133142_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1721679133142_.pic_hd.jpg)

#### Huber's Robust Loss

综合前两个

![1731679133179_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1731679133179_.pic_hd.jpg)

## 图像分类数据集

```
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import dtorch as d2l

d2l.use_svg_display()


def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 数据可视化
# 几行几列现实图片 还有图片的下标
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


batch_size = 256


def get_dataloader_workers():  # @save
    """使用4个进程来读取数据 看cpu"""
    return 4


# 我们可以通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中。

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()  # 预处理
#  train 的意思是:这个数据集是不是训练的数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
#  形成一个数据集
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())


# 读取训练集的时间  常见的性能瓶颈
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'

```

## softMax 代码实现

### 纯手写

````
import torch
from IPython import display
from d2l import dtorch as d2l


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        # 重复了n次
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 数据可视化
class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


#  目标输出函数soft Max的定义 这样子会让里面的每个元素都为正 且每行总和为1
def softmax(X):
    X_exp = torch.exp(X)
    # 每一行求和 从右到左求和 分母 变成一个列向量
    partition = X_exp.sum(1, keepdim=True)
    # 第i行全部元素除以partition的第i个元素
    return X_exp / partition  # 这里应用了广播机制


# 实现softmax回归模型
def net(X):
    # 把X首先reshape一个2d的矩阵，原先是[256,28,28]-1是让程序自己算 实际是256*784的矩阵 256是批量大小 最终变成一个256*10的矩阵
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 计算交叉熵 损失函数
def cross_entropy(y_hat, y):
    #   y是第iy_hat矩阵，的第n个数 log里面就是取y_hat内某个label预测的概率
    return - torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 每行最大
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        #  不计算梯度只做forward pass
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        # 每次拿到一个批量x和y
        for X, y in data_iter:
            # net(X)算评测值
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 定义训练函数

# updater是更新模型参数的常用函数，它接受批量大小作为参数。 它可以是d2l.sgd函数，也可以是框架的内置优化函数。
lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失函数总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            # 自更新下
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            # 自己实现的l求和算梯度
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练的方法
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
# </editor-fold>
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# 读取数据，函数定义见ReadImage
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 拉成一个向量 因为输入需要是一个向量 大小28*28=784，数据集有10个类，所以模型的输出纬度是10
num_inputs = 784
num_outputs = 10
# 行数=784 列数=10 784*10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 每个输出都要有一个偏移，长为10的列向量
b = torch.zeros(num_outputs, requires_grad=True)
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

predict_ch3(net, test_iter)
````

### 使用Torch

```
import torch
from torch import nn
from d2l import dtorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten）变成2d模型，来调整网络输入的形状，作为输入
# Sequential 是构建网络层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# m是当前layer
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# 每一层跑一下这个函数实现初始化？
net.apply(init_weights)
print(net.parameters())
# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 训练优化
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 感知机

Cigama函数有种很多选择，例如下图。输出是一个离散的类，只能输出一个元素只能是是二分类问题，和softmax不同

![1811679406456_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1811679406456_.pic_hd.jpg)

对感知机的更新权重和偏置，相当于批量为1的梯度下降

从几何角度看，就是画一条线，对两种东西进行分类

## 多重感知机

解决xor的问题，生成一个非线性的权重模型，即不随一个变量的变化而变化![1951679918975_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1951679918975_.pic_hd.jpg)

### 单隐藏层单分类

![1961679919961_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1961679919961_.pic_hd.jpg)

如果没有激活函数cita，无论有多少个隐藏层都只是线性函数，永远不能变成非线性

#### sigmoid激活函数

投影范围（0，1）

![1981679920328_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1981679920328_.pic_hd.jpg)



#### Tanh激活函数

投影范围（-1，1）

![1971679920303_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/1971679920303_.pic_hd.jpg)

#### ReLu 激活函数

算的快

![2001679920573_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2001679920573_.pic_hd.jpg)

### 多类分类

和soft Max函数区别不大，多了一个隐藏层，对输出加多了一个softmax函数得到结果

![2011679920748_.pic](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2011679920748_.pic.jpg)

### 多隐藏层

一般来说会每一层逐渐变小，减少信息的损失，但也存在变大的例如CNN，避免overfitting

![2021679921030_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2021679921030_.pic_hd.jpg)

## 多感知机的实现

### 纯手写

```
import torch
from torch import nn
from d2l import dtorch as d2l


# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# 模型的实现

def net(X):
    # 先展平，-1神自己算
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return H @ W2 + b2


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 输入个数，输出个数，隐藏层里面变量的数量
num_inputs, num_outputs, num_hiddens = 784, 10, 256
# 对参数进行初始化
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# 实现 和soft Max没有区别
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)
```

### 使用torch

```
import torch
from torch import nn
from d2l import dtorch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights);
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 模型选择

训练误差：模型在训练数据上的误差

泛化误差：模型在新数据的误差

验证数据集：拿出一些训练的数据进行训练，验证数据集不能和训练数据集混在一起！！！

测试数据集：只用一次的数据集？

### K-则交叉验证

把训练数据分割成K块

```
For i=1,...，K

	使用第i块做验证数据集，其他作为训练数据集
报告K哥验证集误差的平均
常用 
```

## 过拟合和欠拟合

![2031679931905_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2031679931905_.pic_hd.jpg)

## 权重衰退

处理过拟合的方法：即限制权重值的，让w小于某个值

但主要使用柔性限制：pennalty 罚

![2081680103056_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2081680103056_.pic_hd.jpg)

![2091680103072_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2091680103072_.pic_hd.jpg)

入是超参数，控制了w的取值范围

![2101680104076_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2101680104076_.pic_hd.jpg)

n是学习率，后半部分是反向传播，前半部分是权重衰减

## 丢弃法

好的模型对输入数据的扰动鲁棒，使用有噪音的数据等价于Tikhnov正则，丢弃法：在层之间加入噪音

![2191680190179_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2191680190179_.pic_hd.jpg)

期望不发生变化

丢弃法通常作用在隐藏全联接层之中

![截屏2023-03-30 23.33.13](/Users/ha/Library/Application Support/typora-user-images/截屏2023-03-30 23.33.13.png)

通常，我们在测试时不用暂退法。 给定一个训练好的模型和一个新的样本，我们不会丢弃任何节点，因此不需要标准化。

总结 

- 将输出项随机置0控制模型复杂度
- 用于多层感知机的隐藏层输出上
- 丢弃概率是控制模型复杂度的超参数

## 数值的稳定性

![2211680192838_.pic_hd](/Users/ha/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/fe407b7f7a118f0cef48e5bb93bf9196/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/2211680192838_.pic_hd.jpg)

会存在梯度爆炸： 1.5^100 = 4*10^17

​			梯度消失： 0.8^100 =2*10^-10

