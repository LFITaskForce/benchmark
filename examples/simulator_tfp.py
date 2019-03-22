import tensorflow as tf
import tensorflow_probability as tfp

ed = tfp.edward2

tf.enable_eager_execution()


def simulator(inputs):
    w = tf.constant([[0.5, 0.5]])
    n = ed.Normal(0., scale=1., name='noise')
    return tf.matmul(w, inputs, transpose_b=True) + n


inputs = [[1., 2.]]
with ed.tape() as tape:
    x = simulator(inputs)

print(dir(tape))
print(tape)
