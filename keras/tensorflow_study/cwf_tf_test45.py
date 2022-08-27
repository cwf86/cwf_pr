# Chp9
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

(train_images, train_labels),(test_images,test_labels)=keras.datasets.fashion_mnist.load_data()

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
'Shirt', 'Sneaker','Bag', 'Ankle boot']

#print(train_labels,test_labels)
#plt.imshow(train_images[0].reshape(28,28))
#plt.show()

#print(train_images.shape)
#print(train_labels.shape)
#print(test_images.shape)
#print(test_labels.shape)

'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

train_images=train_images/255
test_images=test_images/255

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

