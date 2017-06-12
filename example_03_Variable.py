import tensorflow as tf

# tf 中定义变量的时候一定要使用Variable，即只有定义了变量才是变量
# 其中0 初始值， counter 是变量的name
state = tf.Variable(0, name='counter')
# print(state.name)

# 定义一个常量
one = tf.constant(1)

# 使用tf的add方法
new_value = tf.add(state, one)
# 将 new_value 加载到state上
update = tf.assign(state, new_value)

# 初始化变量 初始化所有变量 tf 1.0+
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 通过run的方式init
    sess.run(init)
    for _ in range(3):
        # 通过run的方式update
        sess.run(update)
        # 通过run的方式才能够输出结果
        print(sess.run(state))





