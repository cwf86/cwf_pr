# Chp9
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


class CNN(object):
    def __init__(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64,(3,3),activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64,(3,3),activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dense(10,activation='softmax'))
        #model.summary()
        self.model=model

class DataSource(object):
    def __init__(self):
        (train_images,train_labels),(test_images,test_labels)=datasets.mnist.load_data()
        train_images=train_images.reshape((60000,28,28,1))
        test_images=test_images.reshape((10000,28,28,1))
        train_images,test_images=train_images/255.0,test_images/255.0
        self.train_images,self.train_labels=train_images,train_labels
        self.test_images,self.test_labels=test_images,test_labels


class Train:
    def __init__(self):
        self.cnn=CNN()
        self.data=DataSource()
    def train(self):
        check_path='./cwf_ckpt/cp-{epoch:04d}.ckpt'
        save_model_cb=tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,
        verbose=1,period=5)#save checkpoint per 5 times
        self.cnn.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_images,self.data.train_labels,
        epochs=5,callbacks=[save_model_cb])

        test_loss,test_acc=self.cnn.model.evaluate(self.data.test_images,self.data.test_labels)
        print("acc:%.4f, total test %d pictures" % (test_acc,len(self.data.test_labels)))

class Predict(object):
    def __init__(self):
        latest=tf.train.latest_checkpoint('./cwf_ckpt')
        self.cnn=CNN()
        if latest == None:
            print('No ckpt can be used')
            exit(-1)
        else:
            self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        img=Image.open(image_path).convert('L')
        #img.show()
        flatten_img=np.reshape(img,(28,28,1))
        x=np.array([1-flatten_img])
        y=self.cnn.model.predict(x)
        print(image_path)
        print(y[0])
        print('       -> Predict digit', np.argmax(y[0]))


if __name__ == '__main__':
#    CNN()

#    test=Train()
#    test.train()

    #maybe because the test picture is made by myself with windows's paint,
    #so the accuracy is not good! 
    test=Predict()
    test.predict('./image_test/test_0.png')
    test.predict('./image_test/test_1.png')
    test.predict('./image_test/test_4.png')
    test.predict('./image_test/test_6.png')
