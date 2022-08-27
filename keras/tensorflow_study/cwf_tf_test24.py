#Chp7
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers
import ssl
import matplotlib.pyplot as plt
from IPython.display import SVG

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

num_words=2000
num_tags=12
num_departments=4

body_input=keras.Input(shape=(None,),name='body')
title_input=keras.Input(shape=(None,),name='title')
tag_input=keras.Input(shape=(num_tags,),name='tag')

body_feat=layers.Embedding(num_words, 64)(body_input)
title_feat=layers.Embedding(num_words, 64)(title_input)

body_feat=layers.LSTM(32)(body_feat)
title_feat=layers.LSTM(128)(title_feat)
features=layers.concatenate([title_feat,body_feat,tag_input])

priority_pred=layers.Dense(1,activation='sigmoid',name='priority')(features)
departments_pred=layers.Dense(num_departments,activation='softmax',name='department')(features)

model=keras.Model(inputs=[body_input,title_input,tag_input],
outputs=[priority_pred,departments_pred])

model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
loss={'priority':'binary_crossentropy',
'department':'categorical_crossentropy'},
loss_weights=[1.,0.2])

title_data=np.random.randint(num_words,size=(1280,10))
body_data=np.random.randint(num_words,size=(1280,100))
tag_data=np.random.randint(2,size=(1280,num_tags)).astype('float32')

priority_label=np.random.random(size=(1280,1))
department_label=np.random.randint(2,size=(1280,num_departments))

history=model.fit(
        {'title':title_data,'body':body_data,'tag':tag_data},
        {'priority':priority_label, 'department':department_label},
        batch_size=32,
        epochs=5
)




