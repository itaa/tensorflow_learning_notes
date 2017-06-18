import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 什么是卷积神经网络 CNN (深度学习)?
# What is Convolutional Neural Networks (deep learning)?

# 卷积神经网络 最常应用于 图片识别
# 卷积 神经网络
# 卷积是说神经网络不在对每一个点的数据进行处理，
# 而是对一个区域进行处理
# Google 自己的 CNN 教程
# https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/63796332430923


# number 1 to 10 image data
# 如果本地没有相应的数据包，会先下载，然后解压数据包
# MNIST_data 是下载数据要保存的位置
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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


# 计算精确度
# compute_accuracy 要使用
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # result 是一个百分比，百分比越高证明越准确
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    # normal 产生随机变量
    # stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
    # of the truncated normal distribution.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return initial


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# x input W is weight
def conv2d(x, W):
    # strides [1, x_movement, y_movement, 1]
    # 前后都要为1
    # VALID SAME padding方式
    # VALID 较小， SAME 和原图一样
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    pass


keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

## conv1 layer

## conv2 layer

## func1 layer

## func2 layer


prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# cross_entropy 分类的时候经常使用softmax + cross_entropy来计算的
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

# important step
# tf.initialize_all_variables() no long valid from
# "2017-03-02", "Use `tf.global_variables_initializer` instead."
init = tf.global_variables_initializer()
sess.run(init)


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

