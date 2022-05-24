import tensorflow as tf
import numpy as np

#Generate data where `y = x*3 + 8`.
xData=np.random.rand(10)
yData=xData*3+8

w = tf.Variable(0.)
b = tf.Variable(0.)

def loss():
    return tf.reduce_mean(tf.square(yData - (w * xData + b))) 

def train(lr):
    with tf.GradientTape() as t: 
        current_loss = loss() 
        lr_weight, lr_bias = t.gradient(current_loss, [w, b]) 
        w.assign_sub(lr * lr_weight)
        b.assign_sub(lr * lr_bias)

for step in range(201):
    train(0.5)
    if step % 20 == 0:
        print(step, w.numpy(), b.numpy())