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


train_images=train_images/255
test_images=test_images/255

model=keras.Sequential(
    [
        layers.Flatten(input_shape=[28,28]),
        layers.Dense(128,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
#model.summary()

model.fit(train_images, train_labels, epochs=5)

predictions = model.predict(test_images)

def plot_image(i,predictions_array,true_label,img):
    prediction_array,true_label,img=predictions_array[i],true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap=plt.cm.binary)
    predicted_label=np.argmax(prediction_array)
    if predicted_label == true_label:
        color='blue'
    else:
        color='red'
    plt.xlabel("{} {:2.0f} ({})".format(class_names[predicted_label],
    100*np.max(prediction_array), class_names[true_label]),color=color)

def plot_value_array(i,predictions_array,true_label):
    prediction_array,true_label=predictions_array[i],true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot=plt.bar(range(10), prediction_array,color='#777777')
    plt.ylim([0,1])
    predicted_label=np.argmax(prediction_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


#only see the image0's result.because [Ankle boot] is the last of class_names,
#so you see the highest column is the last column in the picture.
i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i,predictions,test_labels)
plt.show()





