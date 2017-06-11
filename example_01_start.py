import tensorflow as tf
import numpy as np


# create data
# type is float 32
x_data = np.random.rand(100).astype(np.float32)
# 0.1 is w , 0.3 is b
y_data = x_data * 0.1 + 0.3


# create tensorflow structure start
# create a parameter Variable , 参数范围是 -1.0 到 1.0
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# create 0
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

# 损失函数
loss = tf.reduce_mean(tf.square(y - y_data))

# 优化器 GradientDescentOptimizer 的参数 学习效率learning_rate is 0到1之间
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 优化器 使 loss 最低
train = optimizer.minimize(loss)

# 初始化结构
init = tf.initialize_all_variables()
# create tensorflow structure end

# 激活结构

sess = tf.Session()
# Very important, 激活变量
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 is 0:
        # run Weights 来输出 Weights run biases 来输出 biases
        print(step, sess.run(Weights), sess.run(biases))