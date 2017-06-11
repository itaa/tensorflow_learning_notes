import tensorflow as tf


# matrix1 常量 一行两列的matrix
matrix1 = tf.constant([[3, 3]])
# matrix2 常量 两行一列的matrix
matrix2 = tf.constant([[2],
                       [2]])
# matrix multiply 矩阵乘法
# np.dot(m1, m2)
product = tf.matmul(matrix1, matrix2)


# method 1
sess = tf.Session()
# 每次run的时候tf才会执行以下结构
result = sess.run(product)
print(result)
sess.close()

# method 2 use  "with as"
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)