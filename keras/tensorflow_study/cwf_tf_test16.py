#Chp6
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers
import ssl
import matplotlib.pyplot as plt
from IPython.display import SVG

ssl._create_default_https_context=ssl._create_unverified_context

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

x_train=x_train.reshape((-1,28*28))/255.0
x_test=x_test.reshape((-1,28*28))/255.0

print(x_train.shape, '' ,y_train.shape)
print(x_test.shape, '' ,y_test.shape)

code_dim=32

inputs=layers.Input(shape=(x_train.shape[1],), name='inputs')
code=layers.Dense(code_dim, activation='relu', name='code')(inputs)
outputs=layers.Dense(x_train.shape[1], activation='softmax', 
name='outputs')(code)

auto_encoder=keras.Model(inputs,outputs)
#auto_encoder.summary()

keras.utils.plot_model(auto_encoder, show_shapes=True)
encoder=keras.Model(inputs,code)
keras.utils.plot_model(encoder,show_shapes=True)

decoder_input=keras.Input((code_dim,))
decoder_output=auto_encoder.layers[-1](decoder_input)
decoder=keras.Model(decoder_input,decoder_output)
keras.utils.plot_model(decoder, show_shapes=True)

auto_encoder.compile(optimizer='adam',loss='binary_crossentropy')

history=auto_encoder.fit(x_train,x_train,batch_size=64,epochs=10,
validation_split=0.1)


