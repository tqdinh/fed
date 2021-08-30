# example of calculation 2d convolutions
from numpy import asarray
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from tensorflow.keras import *
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
my_testing_data=np.random.rand(3*3*8)*0.1
my_data=np.reshape(my_testing_data,newshape=(3,3,1,8)).tolist()
data = asarray(my_data)

# create model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(1, (3,3),strides=1))
model.set_weights(data)
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]

my_array=np.array(detector)
print( "....... ",my_array.shape)
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())
# apply filter to input data
yhat = model.predict(data)
for r in range(yhat.shape[1]):
    	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])