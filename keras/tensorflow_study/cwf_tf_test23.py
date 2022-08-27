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

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

encode_input=tf.keras.Input(shape=(28,28,1), name='img')

h1=layers.Conv2D(16,3,activation='relu')(encode_input)
h1=layers.Conv2D(32,3,activation='relu')(h1)
h1=layers.MaxPool2D(3)(h1)
h1=layers.Conv2D(32,3,activation='relu')(h1)
h1=layers.Conv2D(16,3,activation='relu')(h1)
encode_output=layers.GlobalMaxPool2D()(h1)

encode_model=keras.Model(inputs=encode_input, outputs=encode_output,name='encoder')
encode_model.summary()

h2=layers.Reshape((4,4,1))(encode_output)
h2=layers.Conv2DTranspose(16,3,activation='relu')(h2)
h2=layers.Conv2DTranspose(32,3,activation='relu')(h2)
h2=layers.UpSampling2D(3)(h2)
h2=layers.Conv2DTranspose(16,3,activation='relu')(h2)
decode_output=layers.Conv2DTranspose(1,3,activation='relu')(h2)

#in book  inputs=encode_input  is wrong,see line 28
autoencoder=keras.Model(inputs=encode_output, outputs=decode_output,name='autoencoder')
autoencoder.summary()

autoencoder_input=keras.Input(shape=(28,28,1), name='img')
h3=encode_model(autoencoder_input)
autoencoder_output=autoencoder(h3)

autoencoder=keras.Model(inputs=autoencoder_input, outputs=autoencoder_output,name='autodecode')
autoencoder.summary()

