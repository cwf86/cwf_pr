#Chp3
from __future__ import absolute_import,division,print_function
from re import X
import tensorflow as tf
import numpy as np
from numpy import int32

x=tf.ones((2,2))

with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y=tf.reduce_sum(x)#y=x+x+x+x=4x
    z=tf.multiply(y,y)#z=y^2=(4x)^2

dz_dx = t.gradient(z,x)
print(dz_dx)

dz_dy=t.gradient(z,y)
print(dz_dy)

