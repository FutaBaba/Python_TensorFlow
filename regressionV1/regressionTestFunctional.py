import tensorflow.compat.v1 as tf
import  numpy as np

tf.disable_v2_behavior()

xData = np.random.rand(10)
yData = xData * 3 + 8

w = tf.constant(0.)
b = tf.constant(0.)

sess = tf.Session()

def fGradient (sess,learningRate,weight, bias,n):
    if n == 0:
        print (sess.run(weight),sess.run(bias))
        return (sess.run(weight),sess.run(bias))
    else:
        with tf.GradientTape() as tape:
            tape.watch(weight)
            tape.watch(bias)
            loss = tf.reduce_mean(tf.square ((weight * xData + bias) - yData))
        [wGrad,bGrad] = tape.gradient(loss,[weight,bias])
        fGradient(sess,learningRate,weight-wGrad*learningRate,bias-bGrad*learningRate,n-1)

fGradient(sess,tf.constant(0.5),w,b,200)