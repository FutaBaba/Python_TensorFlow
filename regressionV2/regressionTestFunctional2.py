import tensorflow as tf
import numpy as np

#Generate data where `y = x*3 + 8`.
xData=np.random.rand(10)
yData=xData*3+8

w = tf.constant(0.)
b = tf.constant(0.)

def loss(w,b):
    return tf.reduce_mean(tf.square(yData - (w * xData + b)))
    
def train(wb, lr):
    with tf.GradientTape() as t:
        weight = wb[0]
        bias = wb[1]
        newW = tf.Variable(weight)
        newB = tf.Variable(bias)
        current_loss = loss(newW, newB) 
        lr_weight, lr_bias = t.gradient(current_loss, [newW, newB]) 
        newW.assign_sub(lr * lr_weight) 
        newB.assign_sub(lr * lr_bias)
        return [newW.numpy(), newB.numpy()]

def rec(f, wb, lr, n):
    if n == 1:
        return f(wb, lr)
    else:
        return f (rec(f, wb, lr, n - 1), lr)

print(rec(train, [w, b], 0.5, 100))