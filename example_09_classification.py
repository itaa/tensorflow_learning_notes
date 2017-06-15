import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# classification 是一个分类器的问题
# 在之前的例子中都是一些线性回归的问题，或者非线性数据，输入输出就是一(组)对一(组)的数值
# 之前的都是一些连续的数据

# classification是一些分类的问题，输入的是一组数据，输出的结果是一组概率，整组概率的值相加
# 结果为1，可以选择最接近1的值做为输出的结果

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
    #
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # result 是一个百分比，百分比越高证明越准确
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
# 输入时一个28*28像素的图片 28 * 28 = 784
# 输出是一个0到9数字概率的矩阵
# 数据形式是float32
# None表示不规定有多少个sample, 可以为任意多
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
# activation_function 使用的softmax
# softmax 经常用在做分类器的
# prediction 预测值，是一个1*10的概率矩阵
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data
# loss function
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
    # 从下载好的数据中提取 100 个来学习
    # 分批的原因是为了更快的看到结果
    # 而且这种方式也不会比全部加载的效果差

    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

