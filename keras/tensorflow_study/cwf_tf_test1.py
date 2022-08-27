#Chp3
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import numpy as np
from numpy import int32

print("cwf:tf_ver:{%s}" % tf.__version__)


print(tf.add(1,2))
print(tf.add([3,8],[2,5]))
print(tf.square(6))
print(tf.reduce_sum([7,8,9]))
print(tf.square(3)+tf.square(4))

print("\n-------\n")

x=tf.matmul([[3],[6]],[[2]])
print(x)
print(x.shape)
print(x.dtype)

print("\n-------\n")

ndarry = np.ones([2,2])
tensor = tf.multiply(ndarry, 36)
print(tensor)

print(np.add(tensor,1))
print(tensor.numpy())

print("\n-------\n")
# Create Dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([6,5,4,3,2,1])
import tempfile
_, filename = tempfile.mkstemp()
print(filename)

with open(filename, 'w') as f:
    f.write("""L1
L2
L3""")
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)
print('ds_tensors:')
for x in ds_tensors:
    print(x)
print('ds_file:')
for x in ds_file:
    print(x)
