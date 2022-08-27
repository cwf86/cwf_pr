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
import PIL
import imageio
import glob
import time

BUFFER_SIZE=60000
BATCH_SIZE=256
EPOCHS=50
z_dim=100
num_examples_to_generate=16
seed=tf.random.normal([num_examples_to_generate,z_dim])

(train_images,train_labels),(_,_)=keras.datasets.mnist.load_data()

#plt.imshow(train_images[0])
#plt.show()

train_images=train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
train_images=(train_images-127.5)/127.5
train_dataset=tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator():
    generator=keras.Sequential(
        [
            keras.layers.Dense(7*7*256, use_bias=False,input_shape=(100,)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Reshape((7,7,256)),
            keras.layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,
            activation='tanh'),
        ]
    )
    return generator

def make_discriminator():
    discriminator=keras.Sequential(
        [
            keras.layers.Conv2D(64,(5,5),strides=(2,2),padding='same'),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(128,(5,5),strides=(2,2),padding='same'),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(1),
        ]
    )
    return discriminator

'''
g=make_generator()
z=tf.random.normal([1,100])
fake_image=g(z,training=False)
plt.imshow(fake_image[0,:,:,0],cmap='gray')
plt.show()
'''

'''
g=make_generator()
z=tf.random.normal([1,100])
fake_image=g(z,training=False)
d=make_discriminator()
pred=d(fake_image)
print('pred score is:',pred)
'''


g=make_generator()
d=make_discriminator()
z=tf.random.normal([1,100])
fake_image=g(z,training=False)

cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer=keras.optimizers.Adam(1e-4)
d_optimizer=keras.optimizers.Adam(1e-4)

def generator_loss(fake_image):
    return cross_entropy(tf.ones_like(fake_image),fake_image)

def discriminator_loss(fake_image,real_image):
    real_loss=cross_entropy(tf.ones_like(real_image),real_image)
    fake_loss=cross_entropy(tf.ones_like(fake_image),fake_image)
    return real_loss+fake_loss

checkpoint_dir='./training_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')
checkpoint=tf.train.Checkpoint(g_optimizer=g_optimizer,d_optimizer=d_optimizer,g=g,d=d)

@tf.function
def train_one_step(images):
    z=tf.random.normal([BATCH_SIZE,z_dim])
    with tf.GradientTape() as g_tape,tf.GradientTape() as d_tape:
        fake_images=g(z,training=True)
        real_pred=d(images,training=True)
        fake_pred=d(fake_images,training=True)
        g_loss=generator_loss(fake_images)
        d_loss=discriminator_loss(real_pred,fake_pred)
    g_gradients=g_tape.gradient(g_loss,g.trainable_variables)
    d_gradients=d_tape.gradient(d_loss,d.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients,g.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients,d.trainable_variables))

def train(dataset,epochs):
    for epoch in range(epochs):
        start=time.time()
        for image_batch in dataset:
            train_one_step(image_batch)
        generate_and_save_images(g, epoch+1,seed)
        if (epoch +1 %15) == 0:
           checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch+1,time.time()-start))

def generate_and_save_images(model,epoch,test_input):
    predictions=model(test_input,training=False)
    fig=plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,:,:,0]*127.5+127.5,cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

if __name__=='__main__':
    train(train_dataset,EPOCHS)








