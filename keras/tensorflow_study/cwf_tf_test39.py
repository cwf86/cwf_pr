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

model=get_model()

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.RMSprop())
history=model.fit(x_train,y_train,batch_size=64,epochs=1)

model.save_weights('my_model_weights', save_format='tf')

predictions=model.predict(x_test)
first_batch_loss=model.train_on_batch(x_train[:64],y_train[:64])

new_model=get_model()
new_model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.RMSprop())
new_model.load_weights('my_model_weights')

new_predictions=new_model.predict(x_test)

np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)
new_first_batch_loss=new_model.train_on_batch(x_train[:64],y_train[:64])

if first_batch_loss == new_first_batch_loss:
    print('is same!')
else:
    print('not same!')



