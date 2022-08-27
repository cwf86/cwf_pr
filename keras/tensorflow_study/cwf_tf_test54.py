# Chp11
from __future__ import absolute_import, division, print_function
from lib2to3.pgen2.tokenize import tokenize
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


ssl._create_default_https_context=ssl._create_default_https_context

#my VM's memory is not enough for 10000's train_data,so here use 1000!!
NUM_WORDS=1000

(train_data,train_labels),(test_data,test_labels)=keras.datasets.imdb.load_data(
    num_words=NUM_WORDS)

def multi_hot_sequences(sequences,dimension):
    results=np.zeros((len(sequences),dimension))
    for i,word_indices in enumerate(sequences):
        results[i,word_indices]=1.0
    return results

train_data=multi_hot_sequences(train_data,dimension=NUM_WORDS)
test_data=multi_hot_sequences(test_data,dimension=NUM_WORDS)


baseline_model=keras.Sequential(
    [
        layers.Dense(16,activation='relu',input_shape=(NUM_WORDS,)),
        layers.Dense(16,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ]
)

baseline_model.compile(optimizer='adam',loss='binary_crossentropy',
metrics=['accuracy','binary_crossentropy'])

#baseline_model.summary()

baseline_history=baseline_model.fit(train_data,train_labels,epochs=20,batch_size=512,
validation_data=(test_data,test_labels),verbose=2)


small_model=keras.Sequential(
    [
        layers.Dense(4,activation='relu',input_shape=(NUM_WORDS,)),
        layers.Dense(4,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ]
)
small_model.compile(optimizer='adam',loss='binary_crossentropy',
metrics=['accuracy','binary_crossentropy'])

#small_model.summary()
small_history=small_model.fit(train_data,train_labels,epochs=20,batch_size=512,
validation_data=(test_data,test_labels),verbose=2)

big_model=keras.Sequential(
    [
        layers.Dense(512,activation='relu',input_shape=(NUM_WORDS,)),
        layers.Dense(512,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ]
)
big_model.compile(optimizer='adam',loss='binary_crossentropy',
metrics=['accuracy','binary_crossentropy'])

#big_model.summary()
big_history=big_model.fit(train_data,train_labels,epochs=20,batch_size=512,
validation_data=(test_data,test_labels),verbose=2)


dpt_model=keras.Sequential(
    [
        layers.Dense(16,activation='relu',input_shape=(NUM_WORDS,)),
        layers.Dropout(0.5),
        layers.Dense(16,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1,activation='sigmoid')
    ]
)
dpt_model.compile(optimizer='adam',loss='binary_crossentropy',
metrics=['accuracy','binary_crossentropy'])

#dpt_model.summary()
dpt_history=dpt_model.fit(train_data,train_labels,epochs=10,batch_size=512,
validation_data=(test_data,test_labels),verbose=2)


l2_model=keras.Sequential(
    [
        layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),
        activation='relu',input_shape=(NUM_WORDS,)),
        layers.Dropout(0.5),
        layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),
        activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1,activation='sigmoid')
    ]
)
l2_model.compile(optimizer='adam',loss='binary_crossentropy',
metrics=['accuracy','binary_crossentropy'])

#l2_model.summary()
l2_history=l2_model.fit(train_data,train_labels,epochs=10,batch_size=512,
validation_data=(test_data,test_labels),verbose=2)


def plot_history(histories,key='binary_crossentropy'):
    plt.figure(figsize=(16,10))
    for name,history in histories:
        val=plt.plot(history.epoch,history.history['val_'+key],
        '--',label=name.title()+' val')
        plt.plot(history.epoch,history.history[key],color=val[0].get_color(),
        label=name.title()+' train')
    plt.xlabel('epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])

plot_history([('baseline',baseline_history),
('small',small_history),('big',big_history),
('dropout',dpt_history),('l2',l2_history)])
plt.show()

