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

inputs=keras.Input(shape=(784,),name='digits')
x=layers.Dense(64,activation='relu',name='dense_1')(inputs)
x=layers.Dense(64,activation='relu',name='dense_2')(x)
outputs=layers.Dense(10,activation='softmax',name='predictions')(x)

model=keras.Model(inputs,outputs,name='7_5_test')
model.summary()

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.RMSprop())
history=model.fit(x_train,y_train,batch_size=64,epochs=50)
predictions=model.predict(x_test)
print(predictions)

json_config=model.to_json()
reinitialized_model=keras.models.model_from_json(json_config)

new_prediction=reinitialized_model.predict(x_test)
print(new_prediction)

if abs(np.sum(predictions-new_prediction))<0.0001:
    print('is same!', np.sum(predictions-new_prediction))
else:
    print('not same!')




