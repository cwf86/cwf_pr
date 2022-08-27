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

model=get_compiled_model()

class_weight={i:1.0 for i in range(10)}
class_weight[5]=2.0
print(class_weight)
model.fit(x_train,y_train,class_weight=class_weight,batch_size=64,
epochs=4)


model = get_compiled_model()
sample_weight=np.ones(shape=(len(y_train),))
sample_weight[y_train==5]=2.0
model.fit(x_train,y_train,sample_weight=sample_weight,batch_size=64,
epochs=4)

model = get_compiled_model()
sample_weight=np.ones(shape=(len(y_train),))
sample_weight[y_train==5]=2.0

train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset=train_dataset.shuffle(buffer_size=1024).batch(64)

val_dataset=tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_dataset=val_dataset.batch(64)

model.fit(train_dataset, epochs=3,steps_per_epoch=100,
validation_data=val_dataset, validation_steps=3)






