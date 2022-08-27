#Chp5
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers
import ssl
import matplotlib.pyplot as plt

ssl._create_default_https_context=ssl._create_unverified_context

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

print(x_train.shape, '' ,y_train.shape)
print(x_test.shape, '' ,y_test.shape)

x_train=x_train.reshape((-1,28,28,1))
x_test=x_test.reshape((-1,28,28,1))

print(x_train.shape)

model=keras.Sequential()

#the book here is x_train.shape[0],x_train.shape[1],x_train.shape[2]),
#it's wrong!!!
model.add(layers.Conv2D(
input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]), 
filters=32,kernel_size=(3,3),strides=(1,1), padding='valid',
activation='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())

model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


model.compile(optimizer=keras.optimizers.Adam(),
loss=keras.losses.SparseCategoricalCrossentropy(),
metrics=['accuracy'])

#model.summary()

history=model.fit(x_train,y_train,batch_size=64,epochs=5,
validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'],loc='upper left')
plt.show()

