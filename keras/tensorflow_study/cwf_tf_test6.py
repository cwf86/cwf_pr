#Chp3
from __future__ import absolute_import,division,print_function
from re import X
import tensorflow as tf
import numpy as np
from numpy import int32, outer

def f(x,y):
    output = 1.0
    for i in range(y):
        if i>1 and i<5:
            output=tf.multiply(output,x)
    return output

def grad(x,y):
    with tf.GradientTape() as t:
        t.watch(x)
        out=f(x,y)
        return t.gradient(out,x)
x=tf.convert_to_tensor(2.0)
print(grad(x,6))
print(grad(x,5))
print(grad(x,4))
print('--------\n')
x=tf.Variable(1.0)
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y=x*x*x
    dy_dx=t2.gradient(y,x)
    print(dy_dx)
d2y_d2x=t1.gradient(dy_dx,x)
print(d2y_d2x)



