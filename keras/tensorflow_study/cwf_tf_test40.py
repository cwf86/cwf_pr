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
import os

tf.keras.backend.clear_session()

class ThreeLayerMLP(keras.Model):
    def __init__(self,name=None):
        super(ThreeLayerMLP, self).__init__(name=name)
        self.dense_1=layers.Dense(64,activation='relu',name='dense_1')
        self.dense_2=layers.Dense(64,activation='relu',name='dense_2')
        self.pred_layer=layers.Dense(10,activation='softmax',name='predictions')
    def call(self,inputs):
        x=self.dense_1(inputs)
        x=self.dense_2(x)
        return self.pred_layer(x)

def get_model():
    return ThreeLayerMLP(name='3_layer_mlp')

check_path='model.ckpt'
check_dir=os.path.dirname(check_path)

#cp_callback=tf.keras.callbacks.ModelCheckpoint(check_path,
#save_weights_only=True,verbose=1)

cp_callback=tf.keras.callbacks.ModelCheckpoint(check_path,
save_weights_only=True,verbose=1,period=5)

model=get_model()

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

model.compile(loss='sparse_categorical_crossentropy',
optimizer=keras.optimizers.RMSprop())
history=model.fit(x_train,y_train,batch_size=64,epochs=10,callbacks=[cp_callback])

latest=tf.train.latest_checkpoint(check_dir)
#print(latest)

new_model=get_model()
new_model.load_weights(latest)
new_model.compile(loss='sparse_categorical_crossentropy',
optimizer=keras.optimizers.RMSprop())
history=model.fit(x_train,y_train,batch_size=64,epochs=10,callbacks=[cp_callback])


