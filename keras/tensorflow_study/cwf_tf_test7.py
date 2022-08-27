#Chp4
from __future__ import absolute_import,division,print_function
from re import X
import tensorflow as tf
import numpy as np
from numpy import int32, outer


(x_train,y_train),(x_test,y_test)=tf.keras.datasets.boston_housing.load_data()
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

model=tf.keras.Sequential(
    [
    tf.keras.layers.Dense(32,activation='sigmoid',input_shape=(13,)),
    tf.keras.layers.Dense(32,activation='sigmoid'),
    tf.keras.layers.Dense(32,activation='sigmoid'),
    tf.keras.layers.Dense(1)
    ]
)

model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
loss='mean_squared_error', metrics=['mse'])

#run model once
#model.summary()

#run model with train dataset 50 times
model.fit(x_train,y_train,batch_size=50,epochs=50,validation_split=0.1,verbose=1)
result=model.evaluate(x_test,y_test)

#print the model with test dataset result
print(model.metrics_names)
print(result)

