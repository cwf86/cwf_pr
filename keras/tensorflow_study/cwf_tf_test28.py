#Chp7
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers
import ssl
import matplotlib.pyplot as plt
from IPython.display import SVG

tf.keras.backend.clear_session()

class MyLayer(layers.Layer):
    def __init__(self,input_dim=32,uint=32):
        super(MyLayer,self).__init__()
        self.uint=uint

    def build(self,input_shape):
        self.weight=self.add_weight(shape=(input_shape[-1],self.uint),
        initializer=keras.initializers.RandomNormal(),trainable=True)
        
        self.bias=self.add_weight(shape=(self.uint,),
        initializer=keras.initializers.Zeros(),trainable=True)

    def call(self,inputs):
        return tf.matmul(inputs,self.weight)+self.bias

my_layer=MyLayer(3)
x=tf.ones((3,5))
out=my_layer(x)
print(out)

my_layer=MyLayer(3)
x=tf.ones((2,2))
out=my_layer(x)
print(out)




