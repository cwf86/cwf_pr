# Chp8
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from numpy import int32, outer
from tensorflow import keras
from tensorflow.keras import layers
import ssl
import matplotlib.pyplot as plt
from IPython.display import SVG
import os

ssl._create_default_https_context = ssl._create_unverified_context

imdb = keras.datasets.imdb

(train_x, train_y), (test_x, test_y) = keras.datasets.imdb.load_data(num_words=10000)

#print('Training entries: {}, labels: {}'.format(len(train_x), len(train_y)))
#print(train_x[0])
#print('len: ', len(train_x[0]), len(train_x[1]))

word_index=imdb.get_word_index()

word2id = {k:(v+3) for k,v in word_index.items()}
word2id['<PAD>']=0
word2id['<START>']=1
word2id['<UNK>']=2
word2id['<UNUSED>']=3

#print(word2id.items())

id2word={v:k for k,v in word2id.items()}

def get_words(sent_ids):
    return ' '.join([id2word.get(i,'?') for i in sent_ids])

sent=get_words(train_x[0])
#print(sent)

# pad_sequences:add the value of <PAD> on the last of train_x to len 256
train_x = keras.preprocessing.sequence.pad_sequences(
    train_x,value=word2id['<PAD>'],
    padding='post',maxlen=256
)

test_x = keras.preprocessing.sequence.pad_sequences(
    test_x,value=word2id['<PAD>'],
    padding='post',maxlen=256
)

#print(train_x[0])
#print('len: ',len(train_x[0]), len(train_x[1]))

vocab_size=10000
model=keras.Sequential()
model.add(layers.Embedding(vocab_size,16))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
#model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

x_val=train_x[:10000]
x_train=train_x[10000:]
y_val=train_y[:10000]
y_train=train_y[10000:]

#print(x_val)
#print(y_val)

history=model.fit(x_train,y_train,epochs=1,batch_size=512,
validation_data=(x_val,y_val),verbose=1)

result=model.evaluate(test_x,test_y)
print(result)

history_dict=history.history
history_dict.keys()
acc=history_dict['accuracy']
val_acc=history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']
epochs=range(1,len(acc)+1)
'''
plt.plot(epochs,loss,'bo',label='training')
plt.plot(epochs,val_loss,'b',label='validation')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
'''
plt.clf()
plt.plot(epochs,acc,'bo',label='training')
plt.plot(epochs,val_acc,'b',label='validation')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()





