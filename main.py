import tensorflow as tf
from tensorflow.contrib.compiler import xla
import numpy as np

def create_and_run_graph():
  with tf.device('/device:XLA_GPU:0'):
    a = tf.ones([3, 4], tf.int32)
    b = tf.zeros([4, 3], tf.int32)
    c = tf.matmul(a,b)
    sess = tf.Session()
    return sess.run(c)

ab = create_and_run_graph()
print(ab)