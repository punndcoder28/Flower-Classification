


# import tensorflow as tf
# from tensorflow.contrib.compiler import xla
# import numpy as np

# with tf.device('/device:XLA_GPU:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], shape=[3, 2], name='b')
#   c = tf.matmul(a, b)
# # Creates a session with allow_soft_placement and log_device_placement set
# # to True.
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# # Runs the op.
# print(sess.run(c))

