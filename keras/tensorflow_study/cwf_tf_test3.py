#Chp3
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import numpy as np
from numpy import int32

class  MyDense(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(MyDense, self).__init__()
        self.n_outputs = n_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[int(input_shape[-1]), 
        self.n_outputs])
    
    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDense(10)
print(layer(tf.ones([6,5])))
print(layer.trainable_variables)

