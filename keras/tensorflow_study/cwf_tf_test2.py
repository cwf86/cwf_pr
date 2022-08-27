#Chp3
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import numpy as np
from numpy import int32

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(100, input_shape=(None,20))

layer(tf.ones([6,6]))
# print(layer.variables)
print(layer.kernel)
print('------------')
print(layer.bias)

