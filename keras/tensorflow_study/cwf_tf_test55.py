# Chp12
from __future__ import absolute_import, division, print_function
from lib2to3.pgen2.tokenize import tokenize
from pickletools import optimize
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
from sklearn.model_selection import train_test_split

ssl._create_default_https_context=ssl._create_default_https_context

#fuck off the GFW!!
#URL='https://storage.googleapis.com/applied-dl/heart.csv'

heart_csv_path='/home/cwf/tensorflow_study/heart.csv'
dataframe=pd.read_csv(heart_csv_path)

#head=dataframe.head()
#print(head)

train,test=train_test_split(dataframe,test_size=0.2)
train,val=train_test_split(train,test_size=0.2)
#print(len(train),'train examples')
#print(len(val),'validation examples')
#print(len(test),'test examples')

def df_to_dataset(dataframe,shuffle=True,batch_size=32):
    dataframe=dataframe.copy()
    labels=dataframe.pop('target')
    ds=tf.data.Dataset.from_tensor_slices((dict(dataframe),labels))
    if shuffle:
        ds=ds.shuffle(buffer_size=len(dataframe))
    ds=ds.batch(batch_size)
    return ds

batch_size=5
train_ds=df_to_dataset(train,batch_size=batch_size)
val_ds=df_to_dataset(val,shuffle=False,batch_size=batch_size)
test_ds=df_to_dataset(test,shuffle=False,batch_size=batch_size)

'''
for feature_batch,label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch)
'''


example_batch=next(iter(train_ds))[0]
age=tf.feature_column.numeric_column('age')

def print_data(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

#print_data(age)

# (-inf,18),[18,25),[25,30),[30,35),[35,40),[40,50),[50,+inf)
age_buckets=tf.feature_column.bucketized_column(age,boundaries=[18,25,30,35,40,50])
print_data(age_buckets)

#because the col of thal in heart.csv(I get from github) is also number,
#so this example can not run!!
#thal = tf.feature_column.categorical_column_with_vocabulary_list('thal',['fixed','normal','reversible'])
#thal_one_hot = tf.feature_column.indicator_column(thal)
#print_data(thal_one_hot)

#because the col of thal in heart.csv(I get from github) is also number,
#so this example can not run!!
#thal = tf.feature_column.categorical_column_with_vocabulary_list('thal',['fixed','normal','reversible'])
#thal_embedding= tf.feature_column.embedding_column(thal,dimension=8)

#because the col of thal in heart.csv(I get from github) is also number,
#so this example can not run!!
#thal = tf.feature_column.categorical_column_with_vocabulary_list('thal',['fixed','normal','reversible'])
#thal_hashed= tf.feature_column.categorical_column_with_hash_bucket('thal',hash_bucket_size=1000)
#print_data(tf.feature_column.indicator_column(thal_hashed))

#because the col of thal in heart.csv(I get from github) is also number,
#so this example can not run!!
#age_buckets=tf.feature_column.bucketized_column(age,boundaries=[18,25,30,35,40,50])
#thal = tf.feature_column.categorical_column_with_vocabulary_list('thal',['fixed','normal','reversible'])
#crossed_feature=tf.feature_column.crossed_column([age_buckets,thal],hash_bucket_size=1000)

feature_columns=[]

for header in ['age','trestbps','chol','thalach','oldpeak','slope','ca']:
    feature_columns.append(tf.feature_column.numeric_column(header))

age_buckets=tf.feature_column.bucketized_column(age,boundaries=[18,25,30,35,40,50])
feature_columns.append(age_buckets)

#I change the ['fixed','normal','reversible'] to number so the example can run.
#In the heart.csv I get from github,the 'thal' col has  0,1,2,3 four number.
thal = tf.feature_column.categorical_column_with_vocabulary_list('thal',[0,1,2,3])
thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

thal_embedding= tf.feature_column.embedding_column(thal,dimension=8)
feature_columns.append(thal_embedding)

#you can not use categorical_column_with_hash_bucket and crossed_column to same data.
#thal_hashed= tf.feature_column.categorical_column_with_hash_bucket('thal',hash_bucket_size=1000)
#feature_columns.append(tf.feature_column.indicator_column(thal_hashed))

crossed_feature=tf.feature_column.crossed_column([age_buckets,thal],hash_bucket_size=1000)
feature_columns.append(tf.feature_column.indicator_column(crossed_feature))

feature_layer=tf.keras.layers.DenseFeatures(feature_columns)

model=tf.keras.Sequential(
    [
        feature_layer,
        layers.Dense(128,activation='relu'),
        layers.Dense(128,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_ds,validation_data=val_ds,epochs=5)

loss,accuracy=model.evaluate(test_ds)
print('Accuracy,', accuracy)
print('Loss', loss)




