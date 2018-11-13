from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

input = [[1], [2], [3], [4]]
# x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
x = tf.placeholder(tf.float32, shape=[4,1])
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

writer = tf.summary.FileWriter('.', sess.graph)
# writer.add_graph(tf.get_default_graph())

for i in range(1000):
    _, loss_value = sess.run((train, loss), {x: input})
    if (i+1)%100==0: print(loss_value)

writer.close()
print(sess.run(y_pred, {x: input}))