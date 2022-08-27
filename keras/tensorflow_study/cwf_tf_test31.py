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

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

x_val=x_train[-10000:]
y_val=y_train[-10000:]
x_train=x_train[:-10000]
y_train=y_train[:-10000]

class MetricLoggingLayer(layers.Layer):
    def call(self,inputs):
        self.add_metric(keras.backend.std(inputs),name='std_of_activation',
                        aggregation='mean')
        return inputs


inputs=keras.Input(shape=(784,),name='mnist_input')
h1=layers.Dense(64,activation='relu')(inputs)
#h1=MetricLoggingLayer()(h1)
h2=layers.Dense(64,activation='relu')(h1)
outputs=layers.Dense(10,activation='softmax')(h2)

model=keras.Model(inputs,outputs)
# same as h1=MetricLoggingLayer()(h1)
model.add_metric(keras.backend.std(inputs),name='std_of_activation',aggregation='mean')
model.add_loss(tf.reduce_sum(h1)*0.1)

model.compile(optimizer=keras.optimizers.RMSprop(),
loss=keras.losses.SparseCategoricalCrossentropy(),
metrics=[keras.metrics.SparseCategoricalAccuracy()])

history=model.fit(x_train,y_train,batch_size=32,epochs=1,validation_split=0.3)

print('history:')
print(history.history)

result=model.evaluate(x_test,y_test,batch_size=128)
print('evaluate:',result)

pred=model.predict(x_test[:2])
print('predict:',pred)

