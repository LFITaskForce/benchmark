import tensorflow as tf
import tensorflow_probability as tfp

ed = tfp.edward2


def simulator(parameters):
    w = tf.constant([[0.5, 0.5]], name='weights')
    n = ed.Normal(tf.matmul(w, parameters, transpose_b=True), scale=1, name='noise')
    return n


x = simulator([[1., 2.]])

with tf.Session() as sess:
    res = sess.run(x)
