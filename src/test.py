import tensorflow as tf
import numpy as np

pred2 = [[1.0,2.0],[3.0,4.0]]
pred2 = tf.convert_to_tensor(pred2)
pred2_fft = tf.signal.rfft2d(pred2)

with tf.Session() as sess:
    print(sess.run(pred2_fft))
