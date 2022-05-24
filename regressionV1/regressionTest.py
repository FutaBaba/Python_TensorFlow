import tensorflow.compat.v1 as tf
import  numpy as np

tf.disable_v2_behavior()

#Generate data where `y = x*3 + 8`.
xData=np.random.rand(10)
yData=xData*3+8


w = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))

y = w * xData + b
loss=tf.reduce_mean(tf.square(y - yData))

optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)          

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))