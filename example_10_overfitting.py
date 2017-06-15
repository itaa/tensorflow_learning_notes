import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


# 过拟合
# 什么是过拟合？ 过拟合就是机器过于自信，以及到了自负的阶段了
# 自负的坏处就是在自己的小圈子里表现非凡，但是在现实的大圈子里往往处处碰壁
# 自负 = 过拟合

# 结果过拟合的方法
# 1. 增加数据量
# 大多数过拟合的原因是因为数据量太少了
# 如果有足够多的数据就可以减少过拟合
# 2. 运用正规化
# L1, L2.. regularization
# y = Wx :其中 W是机器学习要学到的参数
# 在过拟合中W往往变化过大，为了防止W变化过大，可以通过惩罚参数的方式来减小W的变化
# 原来的误差是 cost = (Wx - real y)^2
# 可以通过将cost的计算公式中加上 W的绝对值的方式来惩罚W过大的情况
# 既：cost = (Wx - real y)^2 + abs(W), 这样使得当W过大的时候cost也随之变大
# cost变大就证明此时的W不是一个很好的参数值。这种是L1的正规化方式

# L2 的方式是将L1中的绝对值换成平方
# 既：cost = (Wx - real y)^2 + (W)^2
# L3，L4...

# 3. Dropout regularization
# 这种方式专门用在神经网络中，既：在训练的过程中，随机忽略一些神经元和神经的连接
# 这个时候神经网络就会变得不完整，用一个不完整的神经网络训练一次，而在下一次
# 又去忽略一些其他的神经元，变成另一个不完整的神经网络，通过这种随机Dropout的规则，
# 就会使得每一次训练和预测的结果都不会特别依赖于某一部分特定的神经元。


# load data 加载数据
digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)


# 添加神经层
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # Weights define
    # 权重，尽量要是一个随机变量
    # 随机变量在生成初始变量的时候比全部为零效果要好的很多
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            # histogram 直方图；柱状图 总结weights
            tf.summary.histogram(layer_name + '/weights', Weights)

        # biases define
        # 偏值项，是一个列表，不是矩阵，默认设置为0 + 0.1
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.add(tf.zeros([1, out_size]), 0.1), name='b')
            # histogram 直方图；柱状图 总结biases
            tf.summary.histogram(layer_name + '/biases', biases)
        # W * x + b
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        # 如果activation_function是空的时候就表示是一个线性关系直接放回即可
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            # histogram 直方图；柱状图 总结biases
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

# placeholder
# 输入是64个单位8*8 输出是10个单位[0,1,2,.....9]
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 100, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 100, 10, 'l2', activation_function=tf.nn.softmax)

# the error between prediction and real data
# loss function
# cross_entropy 分类的时候经常使用softmax + cross_entropy来计算的
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()

