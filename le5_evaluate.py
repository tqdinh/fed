from typing import Any
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from tensorflow.keras import *
from tensorflow.python.ops.gen_array_ops import reverse
import torch as nn
from multiprocessing.context import Process
from multiprocessing import Queue
import multiprocessing.managers as m
from federated_config import *

from numpy import asarray



from datetime import datetime
from federated_config import *
from read_write_file import *

import os
from datetime import datetime
import shutil

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)

test_x = tf.expand_dims(test_x, 3)


val_x = train_x[:5000]
val_y = train_y[:5000]

train_x=train_x[5000:]
train_y=train_y[5000:]

for k in range(0,10):
    model_i = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28, 1)),
    keras.layers.Conv2D(8, kernel_size=3, strides=1  ), #C1
    keras.layers.MaxPooling2D(), #S2
    keras.layers.Flatten(), #Flatten
    keras.layers.Dense(10, activation='softmax') #Output layer
])

    inti_model=model_i.get_weights()
    # path="2021-08-30 12:29:53.161578/0.0/round_{0}".format(100)
    # www=read_list(path)

    # new_www=[]
    # shape=[(3,3,1,8),(8,),(1352,10),(10,)]
    # for j in range(len(www)):
    #     w=np.reshape(np.array(www[j]),newshape=shape[j])
    #     new_www.append(w)


    # model_i.set_weights(new_www)

    model_i.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    history=model_i.fit(train_x,train_y,epochs=1 )
    new_w=model_i.get_weights()
    delta=np.array(new_w)-np.array(inti_model)
    print(delta)
