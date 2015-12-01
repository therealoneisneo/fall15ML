import numpy as np
import DataProcess as dp
import tensorflow as tf


a = tf.Variable(tf.random_normal([20,50], stddev = 1.2), name = "weights")

print a



# print a 