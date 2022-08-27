# Chp8
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers
import ssl
import matplotlib.pyplot as plt
from IPython.display import SVG
import os
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences

ssl._create_default_https_context = ssl._create_unverified_context

num_features=3000
sequence_length=300
embedding_dimension=100

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)

x_train=pad_sequences(x_train,maxlen=sequence_length)
x_test=pad_sequences(x_test,maxlen=sequence_length)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)


def imdb_cnn():
    model=keras.Sequential([
        layers.Embedding(input_dim=num_features,
        output_dim=embedding_dimension,input_length=sequence_length),

        layers.Conv1D(filters=50,kernel_size=5,strides=1,padding='valid'),
        layers.MaxPool1D(2,padding='valid'),
        layers.Flatten(),
        layers.Dense(10,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'])

    return model

model=imdb_cnn()
#model.summary()

history=model.fit(x_train,y_train,batch_size=64,
epochs=5,validation_split=0.1)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend(['training','validation'], loc='upper left')
plt.show()

sys.exit(0)
