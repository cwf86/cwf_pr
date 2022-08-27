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

ssl._create_default_https_context = ssl._create_unverified_context

dataset_path=keras.utils.get_file('auto-mpg.data',
'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')

column_names=['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration',
'Model Year','Origin']

raw_dataset=pd.read_csv(dataset_path,names=column_names,na_values="?",comment='\t',
sep=" ",skipinitialspace=True)

#tail=raw_dataset.tail()
#print(tail)

#All=raw_dataset
#print(All)

#Nan=raw_dataset.isna().sum()
#print(Nan)

dataset=raw_dataset.copy()
dataset=dataset.dropna()

origin=dataset.pop('Origin')
dataset['USA']=(origin==1)*1.0
dataset['Europe']=(origin==2)*1.0
dataset['Japan']=(origin==3)*1.0

#print(dataset)
train_dataset=dataset.sample(frac=0.8,random_state=0)
test_dataset=dataset.drop(train_dataset.index)
train_stats=train_dataset.describe()
test_stats=test_dataset.describe()

'''
print('train_stats')
print(train_stats)
print('\n')

print('test_stats')
print(train_stats)
print('\n')
'''

train_stats.pop('MPG')
train_stats=train_stats.transpose()
#print(train_stats)

train_labels=train_dataset.pop('MPG')
test_labels=test_dataset.pop('MPG')

def norm(x):
    return (x-train_stats['mean'])/train_stats['std']
normed_train_data=norm(train_dataset)
normed_test_data=norm(test_dataset)

'''
print('normed_train_data')
print(normed_train_data)
print('\n')
print('normed_test_data')
print(normed_test_data)
'''

def build_model():
    model=keras.Sequential(
        [
            layers.Dense(64,activation='relu',input_shape=[len(train_dataset.keys())]),
            layers.Dense(64,activation='relu'),
            layers.Dense(1)
        ]
    )
    model.compile(loss='mse',optimizer=keras.optimizers.RMSprop(0.001),metrics=['mae','mse'])
    return model

model=build_model()
#model.summary()
early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
history=model.fit(normed_train_data,train_labels,epochs=1000,validation_split=0.2,verbose=1,
callbacks=[early_stop])
test_result=model.predict(normed_test_data)
#print('\ntest_result:',test_result)

def plot_history(history):
    hist=pd.DataFrame(history.history)
    hist['epoch']=history.epoch
    plt.figure('MAE---MSE',figsize=(8,4))
    plt.subplot(1,2,1)
    plt.xlabel('epoch')
    plt.ylabel('Mean Absolute Error (MPG)')
    plt.plot(
        hist['epoch'],hist['mae'],
        label='train_error')
    plt.plot(
        hist['epoch'],hist['val_mae'],
        label='val_error')
    plt.ylim([0,5])
    plt.legend()
    plt.subplot(1,2,2)
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error ($MPG^2$)')
    plt.plot(
        hist['epoch'],hist['mse'],
        label='train error')
    plt.plot(
        hist['epoch'],hist['val_mse'],
        label='train error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

#plot_history(history)

test_predictions=model.predict(normed_test_data).flatten()

plt.figure('Prediction & TrueValues --- Error',figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(test_labels,test_predictions)
plt.xlabel('true values(MPG)')
plt.ylabel('predictions(MPG)')

plt.axis('equal')
plt.axis('square')

plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
#draw line (-100,-100),(100,100)
_=plt.plot([-100,100],[-100,100],'r-')
'''
error=test_predictions-test_labels
plt.subplot(1,2,2)
plt.hist(error,bins=25)
plt.xlabel(" prediction error(MPG)")
_=plt.ylabel(" count")
'''
plt.show()

