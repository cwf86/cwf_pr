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

inputs=tf.keras.Input(shape=(784,), name='img')

h1=layers.Dense(32,activation='relu')(inputs)
h2=layers.Dense(32,activation='relu')(h1)

outputs=layers.Dense(10,activation='softmax')(h2)

model=tf.keras.Model(inputs=inputs, outputs=outputs)
#model.summary()

keras.utils.plot_model(model,'mnist_model.png')
keras.utils.plot_model(model,'model_info.png',show_shapes=True)

model.compile(optimizer=keras.optimizers.RMSprop(),
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

history=model.fit(x_train,y_train,batch_size=64, epochs=5,
validation_split=0.2)

test_scores=model.evaluate(x_test,y_test,verbose=0)
print('test loss:',test_scores[0])
print('test acc:',test_scores[1])

