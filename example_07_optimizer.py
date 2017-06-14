import tensorflow as tf


# optimizer
# 优化器 Optimizer 加速神经网络训练 (深度学习) Speed up neural network training process (deep learning)
# SGD Stochastic Gradient Descent
# Stochastic adj. [数] 随机的；猜测的
# 随机梯度下降
# Momentum, 惯性原则：在走的路上动手脚，将平地换成一个下坡，就可以避免走很多弯路
# m = b1 * m - Learning rate * dx
# W += m
# AdaGrad, 错误方向的阻力：这种方式是在学习率上动动手脚，给他一双特殊的鞋子，使得他在走弯路的时候会不舒服
# 这样他就会走你想让他走的路了
# v += dx^2
# W += -Learning rate * dx / √V (v的开方)
# RMSProp
# 结合以上两种方式, 但是并没有把Momentum完全合并，还缺少了Momentum的 【- Learning rate * dx】
# v = b1 * v + (1 - b1) * dx^2
# W += -Learning rate * dx / √V (v的开方)
# Adam 是补上RMSProp中缺少的Momentum的 【- Learning rate * dx】
# m = b1 * m - Learning rate * dx ---> Momentum
# W += -Learning rate * dx / √V (v的开方) --->  AdaGrad
# 实验表明，大多数时候使用Adam都能又快又好的达到目标




