# Chp13
from __future__ import absolute_import, division, print_function
from lib2to3.pgen2.tokenize import tokenize
from pickletools import optimize
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers,datasets,models
import tensorflow_datasets as tfds
import ssl
import matplotlib.pyplot as plt
from IPython.display import SVG
import os
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import PIL
import imageio
import glob
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
#from keras.optimizers import SGD


X=np.linspace(-2*np.pi, 2*np.pi,300)#-360(-2pi)~360(2pi)
X=np.reshape(X,[X.__len__(),1])
nosie=np.random.rand(X.__len__(),1)*0.1
Y=np.sin(X)+nosie
#plt.scatter(X,Y)
#plt.show()

X_train,Y_train=X[:160],Y[:160]
X_test,Y_test=X[160:],Y[160:]

#output_dim and input_dim and init is not used in tf2.8
model=models.Sequential(
    [
        layers.Dense(100,kernel_initializer='random_uniform',activation='relu',
        input_shape=(1,)),
        layers.Dense(50,activation='relu'),
        layers.Dense(1,activation='tanh')
    ]
)

model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
#model.summary()

history=model.fit(X,Y,verbose=1,epochs=100,batch_size=10,shuffle=True,
validation_data=(X_test,Y_test))

Y_pred=model.predict(X_test,batch_size=1)
plt.scatter(X_test,Y_test)
#plt.plot(X_test,Y_pred,'b-')
plt.plot(X_test,Y_pred,'r.')
plt.show()




