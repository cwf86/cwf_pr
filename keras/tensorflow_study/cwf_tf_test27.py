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
        
        self.weight=self.add_weight(shape=(input_dim,uint),
        initializer=keras.initializers.RandomNormal(),trainable=True)
        
        self.bias=self.add_weight(shape=(uint,),
        initializer=keras.initializers.Zeros(),trainable=True)
    def call(self,inputs):
        return tf.matmul(inputs,self.weight)+self.bias

class AddLayer(layers.Layer):
    def __init__(self,input_dim=32):
        super(AddLayer,self).__init__()
        self.sum=self.add_weight(shape=(input_dim,),
        initializer=keras.initializers.Zeros(), trainable=False)
    def call(self,inputs):
        self.sum.assign_add(tf.reduce_sum(inputs,axis=0))
        return self.sum


x=tf.ones((3,3))
my_layer=AddLayer(3)
out=my_layer(x)
print(out.numpy())
out=my_layer(x)
print(out.numpy())
print('weight:',my_layer.weights)
print('non-trainable weight:',my_layer.non_trainable_weights)
print('trainable weight:',my_layer.trainable_weights)



