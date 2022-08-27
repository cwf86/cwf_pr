# Chp8
from __future__ import absolute_import, division, print_function
from lib2to3.pgen2.tokenize import tokenize
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import ssl
import matplotlib.pyplot as plt
from IPython.display import SVG
import os
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences


ssl._create_default_https_context = ssl._create_unverified_context

# this action is failed because we can't access www.googleapis.com,
# so the chapter 8.3 can't be studied. 
dataset,info=tfds.load('imdb_reviews/subwords8k',with_info=True,
as_supervised=True)

train_dataset,test_dataset=dataset['train'], dataset['test']
tokenizer = info.fetures['text'].encoder

print('vocabulary size:',tokenizer.vocab_size)


