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

print(tf.keras.__version__)

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

x_val=x_train[-10000:]
y_val=y_train[-10000:]
x_train=x_train[:-10000]
y_train=y_train[:-10000]

def get_compiled_model():
    inputs=keras.Input(shape=(784,),name='mnist_input')
    h1=layers.Dense(64,activation='relu')(inputs)
    h2=layers.Dense(64,activation='relu')(h1)

    outputs=layers.Dense(10,activation='softmax')(h2)
    model=keras.Model(inputs,outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs):
        self.losses=[]
    def on_epoch_end(self,batch,logs):
        self.losses.append(logs.get('loss'))
        print('\nloss:',self.losses[-1])

model=get_compiled_model()

model.fit(x_train,y_train,epochs=20,batch_size=64,
callbacks=[LossHistory()], validation_split=0.2)















