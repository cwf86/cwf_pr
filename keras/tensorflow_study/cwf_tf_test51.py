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


y_train=df_train.pop('survived')
y_test=df_test.pop('survived')

#pd.concat([df_train,y_train],axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
#pd.concat([df_train,y_train],axis=1).groupby('class').survived.mean().plot(kind='pie').set_xlabel('% survive')
#plt.show()

'''
def calc_age_section(n,lim):
    return '[%.f,%.f)' % (lim*(n//lim),lim*(n//lim)+lim)

addone=pd.Series([calc_age_section(s,10) for s in df_train.age])
df_train['ages'] = addone
#pd.concat([df_train,y_train],axis=1).groupby('ages').survived.mean().plot(kind='barh').set_xlabel('% survive')
pd.concat([df_train,y_train],axis=1).groupby('ages')['survived'].mean().plot(kind='barh').set_xlabel('% survive')
plt.show()
'''

CATEGORICAL_COLUMNS=['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
NUMERIC_COLUMNS=['age','fare']

def one_hot_cat_column(feature_name,vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name,vocab)
    )

feature_columns=[]

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary=df_train[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name,vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name,dtype=tf.float32))

NUM_EXAMPLES=len(y_train)

def make_input_fn(X,y,n_epochs=None,shuffle=True):
    def input_fn():
        dataset=tf.data.Dataset.from_tensor_slices((dict(X),y))

        if shuffle:
            dataset=dataset.shuffle(NUM_EXAMPLES)
        dataset=dataset.repeat(n_epochs)
        dataset=dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn

train_input_fn=make_input_fn(df_train,y_train)
eval_input_fn=make_input_fn(df_test,y_test,shuffle=False,n_epochs=1)

linear_est=tf.estimator.LinearClassifier(feature_columns)
linear_est.train(train_input_fn,max_steps=100)

result=linear_est.evaluate(eval_input_fn)
print('-------------------')
print(pd.Series(result))

'''
pred_dicts1=list(linear_est.predict(eval_input_fn))
probs1=pd.Series([pred['probabilities'][1] for pred in pred_dicts1])
plt.figure(figsize=(7,5))
probs1.plot(kind='hist',bins=20,title="Linear-Est predicted probabilities")
plt.show()
'''

from sklearn.metrics import roc_curve

def plot_roc(probs,title):
    fpr,tpr,_=roc_curve(y_test,probs)
    plt.plot(fpr,tpr)
    plt.title(title)
    plt.xlabel('false postitive(FP) rate')
    plt.ylabel('true postitive(TP) rate')
    plt.xlim(0,)
    plt.ylim(0,)

pred_dicts1=list(linear_est.predict(eval_input_fn))
probs1=pd.Series([pred['probabilities'][1] for pred in pred_dicts1])
plt.figure(figsize=(7,5))
plot_roc(probs1,'Linear-est ROC')
plt.show()




