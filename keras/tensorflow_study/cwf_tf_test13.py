#Chp4
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

ssl._create_default_https_context = ssl._create_unverified_context

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train=x_train.reshape([x_train.shape[0],-1])
x_test=x_test.reshape([x_test.shape[0],-1])

#print(x_train.shape,' ',y_train.shape)
#print(x_test.shape,' ',y_test.shape)


def mlp_model():
    model=keras.Sequential(
        [
            layers.Dense(64,activation='relu',
            input_shape=(784,)),
            layers.Dropout(0.2),
            layers.Dense(64,activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64,activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10,activation='softmax'),
        ]
    )
    model.compile(optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])
    return model

model1 = KerasClassifier(model=mlp_model, epochs=30,verbose=0)#build_fn is renameing to model,epoch=100 is too slow...(my vm is low spec)
model2 = KerasClassifier(model=mlp_model, epochs=30,verbose=0)
model3 = KerasClassifier(model=mlp_model, epochs=30,verbose=0)

ensemble_clf=VotingClassifier(estimators=[('model1',model1),('model2',model2),
                                          ('model3',model3)],voting='soft')

#ensemble_clf.summary()

ensemble_clf.fit(x_train,y_train)
y_pred=ensemble_clf.predict(x_test)
print('acc: ', accuracy_score(y_pred, y_test))

print(y_pred)
print('---------------')
print(y_test)



