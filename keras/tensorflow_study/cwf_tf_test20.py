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

callbacks=[
    keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-2,
    patience=2,
    verbose=1)
]
model.fit(x_train,y_train,epochs=20,batch_size=64,
callbacks=callbacks, validation_split=0.2)

model=get_compiled_model()
check_call_back=keras.callbacks.ModelCheckpoint(
    filepath='mymodel_{epoch}.h5',
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)
model.fit(x_train,y_train,epochs=3,batch_size=64,
callbacks=[check_call_back], validation_split=0.2)

inttial_learning_rate=0.1
lr_schedule=keras.optimizers.schedules.ExponentialDecay(
    inttial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

optimizer=keras.optimizers.RMSprop(learning_rate=lr_schedule)

tensorboard_cbk=keras.callbacks.TensorBoard(log_dir='./7_1_test_log')
model.fit(x_train,y_train,epochs=5,batch_size=64,
callbacks=[tensorboard_cbk], validation_split=0.2)















