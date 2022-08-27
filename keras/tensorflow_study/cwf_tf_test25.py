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

ssl._create_default_https_context=ssl._create_unverified_context

inputs=keras.Input(shape=(32,32,3),name='img')

h1=layers.Conv2D(32,3,activation='relu')(inputs)
h1=layers.Conv2D(64,3,activation='relu')(h1)
block1_out=layers.MaxPooling2D(3)(h1)

h2=layers.Conv2D(64,3,activation='relu',padding='same')(block1_out)
h2=layers.Conv2D(64,3,activation='relu',padding='same')(h2)
block2_out=layers.add([h2,block1_out])

h3=layers.Conv2D(64,3,activation='relu',padding='same')(block2_out)
h3=layers.Conv2D(64,3,activation='relu',padding='same')(h3)
block3_out=layers.add([h3,block2_out])

h4=layers.Conv2D(64,3,activation='relu',padding='same')(block3_out)
h4=layers.GlobalMaxPool2D()(h4)
h4=layers.Dense(256,activation='relu')(h4)
h4=layers.Dropout(0.5)(h4)
outputs=layers.Dense(10,activation='softmax')(h4)

model=keras.Model(inputs,outputs)
model.summary()

(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()
x_train=x_train.astype('float32')/255
x_test=y_train.astype('float32')/255
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss='categorical_crossentropy',metrics=['acc'])
model.fit(x_train,y_train,batch_size=64,epochs=1,validation_split=0.2)




