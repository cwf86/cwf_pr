#Chp5
from __future__ import absolute_import,division,print_function
from re import X
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
'''
tf.keras.wrappers.scikit_learn is not supported(tf>2.2),use scikeras.wrappers instead.
see https://github/adriangb/scikeras
and https://www.adriangb.com/scikeras/stable/migration.html
'''
from scikeras.wrappers import KerasClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import ssl
import matplotlib.pyplot as plt

ssl._create_default_https_context=ssl._create_unverified_context

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
#x_train=x_train.reshape([x_train.shape[0],-1])
#x_test=x_test.reshape([x_test.shape[0],-1])

plt.imshow(x_train[0])
plt.show()

