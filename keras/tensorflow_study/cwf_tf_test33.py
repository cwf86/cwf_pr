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

class ActivityRegularizationLayer(layers.Layer):
    def call(self,inputs):
        self.add_loss(1e-2*tf.reduce_sum(inputs))
        return inputs

inputs=keras.Input(shape=(784,),name='digits')
x=layers.Dense(64,activation='relu',name='dense_1')(inputs)
x=ActivityRegularizationLayer()(x)
x=layers.Dense(64,activation='relu',name='dense_2')(x)
outputs=layers.Dense(10,activation='softmax',name='predictions')(x)

model=keras.Model(inputs,outputs)
optimizer=keras.optimizers.SGD(learning_rate=1e-3)
loss_fn=keras.losses.SparseCategoricalCrossentropy()
train_acc_metric=keras.metrics.SparseCategoricalAccuracy()
val_acc_metric=keras.metrics.SparseCategoricalAccuracy()

logits=model(x_train[:64])
print(model.losses)

logits=model(x_train[:64])
logits=model(x_train[64:128])
logits=model(x_train[128:192])

batch_size=64
train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset=train_dataset.shuffle(buffer_size=1024).batch(batch_size)

for epoch in range(3):
    print('Start of epoch:',epoch)
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits=model(x_batch_train)
            loss_value=loss_fn(y_batch_train, logits)
            loss_value+=sum(model.losses)
        grads=tape.gradient(loss_value,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        train_acc_metric(y_batch_train, logits)
        if step % 200 == 0:
            print('Training loss(for one epoch) as step %s: %s' % (step,float(loss_value)))
            print('Seen so far: %s samples' % ((step+1)*64))


