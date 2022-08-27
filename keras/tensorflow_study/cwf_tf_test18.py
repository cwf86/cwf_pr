#Chp6
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers
import ssl
import matplotlib.pyplot as plt
from IPython.display import SVG

ssl._create_default_https_context=ssl._create_unverified_context

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

x_train=tf.expand_dims(x_train.astype('float32'),-1)/255
x_test=tf.expand_dims(x_test.astype('float32'),-1)/255

print(x_train.shape, '' ,y_train.shape)
print(x_test.shape, '' ,y_test.shape)

inputs=layers.Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),
name='inputs')
print('inputs shape:',inputs.shape)

code=layers.Conv2D(16,(3,3),activation='relu',padding='same')(inputs)
code=layers.MaxPool2D((2,2),padding='same')(code)
print('code shape:',code.shape)

decoded=layers.Conv2D(16,(3,3),activation='relu',padding='same')(code)
decoded=layers.UpSampling2D((2,2))(decoded)
print('decoded shape:',decoded.shape)

outputs=layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(decoded)
print('outputs shape:',outputs.shape)



auto_encoder=keras.Model(inputs,outputs)
auto_encoder.compile(optimizer=keras.optimizers.Adam(),
loss=keras.losses.BinaryCrossentropy())
keras.utils.plot_model(auto_encoder,show_shapes=True,to_file=r'./auto_encoder.png')

early_stop=keras.callbacks.EarlyStopping(patience=2,monitor='loss')
auto_encoder.fit(x_train,x_train,batch_size=64,epochs=1,validation_split=0.1,
validation_freq=10,callbacks=[early_stop])

decoded=auto_encoder.predict(x_test)

n=5
for i in range(n):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(tf.reshape(x_test[i+1],(28,28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax=plt.subplot(2,n,n+i+1)
    plt.imshow(tf.reshape(decoded[i+1],(28,28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
