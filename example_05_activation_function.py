import tensorflow as tf
import numpy as np


# Linear Nonlinear
# 线性 非线性
# y = Wx 线性函数
# y = AF(Wx) 激励函数
# AF()
# relu sigmoid tanh
# 可以创造自己的激励函数，但是要求激励函数必须是可以微分的
# 因为在误差反向传播的时候，只有可以微分的函数才能够将误差传递回去
# 切记在多层网络的时候不能随便选择激励函数， 因为如果选择不对会造成梯度爆炸和梯度消失的问题

# CNN（Convolutional Neural Network）
# 卷积神经网络推荐使用 relu
# RNN (Recurrent Neural Network)
# 循环神经网络推荐使用 relu or tanh

# 激励函数 activation function 应该放在layer将要输出的时候

# https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/activation_functions_

# relu: 当x<0时候，y=0, 当x>0 时候 y = Wx + b
# softplus 用作分类器
# 每一个activation function 都有自己的适用之处

# 添加神经层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # Weights define
    # 权重，尽量要是一个随机变量
    # 随机变量在生成初始变量的时候比全部为零效果要好的很多
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases define
    # 偏值项，是一个列表，不是矩阵，默认设置为0 + 0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # W * x + b
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 如果activation_function是空的时候就表示是一个线性关系直接放回即可
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs




# x_data 从-1到1的区间有300个单位
# [:, np.newaxis] 加上一个维度，有300行，有300个例子
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise is a [0, 0.05]之间的一个随机数
# 加上一个noise使得更像真实的数据
noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = x_data^2 -0.5 + noise
y_data = np.square(x_data) - 0.5 + noise

# define xs ys
# placeholder
# 这里的None表示无论输入多少个sample都可以
# 是一个多行单列的矩阵，或者说是一个列表
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 定义隐藏层 define hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 定义输出层 define output layer
# prediction layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# loss function
# 损失函数 axis is new reduction_indices
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    axis=[1]))

# 进行训练
# 设置学习速率为0.1 通常设置为小于1的数字
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 对所有的变量进行初始化
# this a very important step
# 如果不进行初始化后续将无法运行
# initialize_all_variables  deprecated("2017-03-02", "Use `tf.global_variables_initializer` instead.")
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


for i in range(1000):
    # train_step 训练
    # 其中的feed_dict is input data
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 is 0:
        # run loss, 只要是使用了placeholder的地方都要使用feed_dict传入
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))



