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

NUM_WORDS=10000
(train_data,train_labels),(test_data,test_labels)=keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences,dimension):
    results=np.zeros((len(sequences),dimension))
    for i,word_indices in enumerate(sequences):
        results[i,word_indices]=1.0
    return results

train_data=multi_hot_sequences(train_data,dimension=NUM_WORDS)
test_data=multi_hot_sequences(test_data,dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.show()




