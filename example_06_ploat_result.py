import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
# noise
# 加上一个noise使得更像真实的数据
noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = x_data^2 -0.5
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

# 生成一个图片框
fig = plt.figure()
# 1, 1, 1 表示一行一列 第一个
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
# show 的时候把程序暂停看了
# 新版可以使用plt.ion()的方式来继续划线
# 旧版本中 使用plt.show(block=False)
plt.ion()
plt.show()
#
for i in range(2000):
    # train_step 训练
    # 其中的feed_dict is input data
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 is 0:
        # run loss, 只要是使用了placeholder的地方都要使用feed_dict传入
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

        try:
            # remove line
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        # x, y, 红线, 线宽=5
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        # 暂停0.1秒
        plt.pause(0.1)



