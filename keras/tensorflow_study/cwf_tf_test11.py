#Chp4
from __future__ import absolute_import,division,print_function
from re import X
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train=x_train.reshape([x_train.shape[0],-1])
x_test=x_test.reshape([x_test.shape[0],-1])

print(x_train.shape,' ',y_train.shape)
print(x_test.shape,' ',y_test.shape)


model=keras.Sequential(
    [
        layers.Dense(64,activation='relu',kernel_initializer='he_normal',
        input_shape=(784,)),
        layers.BatchNormalization(),
        layers.Dense(64,activation='relu',kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dense(64,activation='relu',kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dense(10,activation='softmax'),
    ]
)

model.compile(optimizer=keras.optimizers.SGD(),
loss=keras.losses.SparseCategoricalCrossentropy(),
metrics=['accuracy'])

#model.summary()

history=model.fit(x_train,y_train,batch_size=256,
epochs=100,validation_split=0.3,verbose=0)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'],loc='upper left')
plt.show()

print('---------\n')
#some web say history.history['acc'] and history.history['val_acc'],
#but it's wrong!!!
print(history.history)
print('---------\n')

result=model.evaluate(x_test,y_test)
print(result)

