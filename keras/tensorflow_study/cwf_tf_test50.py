# Chp10
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

df_train=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
df_test=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

#head=df_train.head()
#print(head)

#tail=df_train.tail()
#print(tail)

#info=df_train.info()
#print(info)

#describe=df_train.describe()
#print(describe)

#df_train.age.hist(bins=20)
#plt.show()

#df_train.sex.value_counts().plot(kind='barh')
#df_train['sex'].value_counts().plot(kind='pie')
#plt.show()

#'class' is a key word in python so next line 'df_train.class' is not work
#df_train.class.value_counts().plot(kind='barh')
#df_train['class'].value_counts().plot(kind='barh')
#plt.show()

#same as df_train['embark_town']
df_train.embark_town.value_counts().plot(kind='barh')
plt.show()






