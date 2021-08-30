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


class my_client(Process):
    def __init__(self,_w_queue):
        super(my_client, self).__init__()
        self.train_x =Any
        self.train_y=Any
        self.w=Any
        self.history=Any
        self.w_queue=_w_queue
    def get_sample_count(self):
        return len(self.train_y)
        
        
    def set_train_x_train_y(self,_train_x,_train_y):
        self.train_x=_train_x
        self.train_y=_train_y
    
    

    def train(self):
        model_i=keras.models.Sequential()

        
        model_i.add(keras.layers.InputLayer(input_shape=(28, 28, 1)))
        model_i.add(keras.layers.Conv2D(8, kernel_size=3, strides=1  ))
        model_i.add(keras.layers.MaxPooling2D())
        model_i.add(keras.layers.Flatten())
        model_i.add(keras.layers.Dense(10, activation='softmax'))

        
        model_i.set_weights(self.w)
        
        model_i.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        self.history=model_i.fit(self.train_x, self.train_y, epochs=1)
        tmpx=model_i.get_weights()
        self.w_queue.put(tmpx)
        

    def get_weight(self):
        return np.copy(self.w)
    
    def get_history(self):
        return self.history
   
    def set_weight(self,weight):
        self.w=weight

    def run(self):    
        self.train()
        #print("hello")
        

class Federated:
    def __init__(self,_C,_K):
        self.C =_C# fration of client each round
        self.KK =_K# num ber of client
        self.m =0#max(C*K,1)
        self.St =0 #random set of client 
        self.w=[]
        self.client_model=[]
        self.client_nk=[]
        self.total_sample=0
        self.tmp_count_client_updated=0

    def getNumberOfClient(self):
        return self.St
   
    def set_weights(self,weight):
        self.w=np.copy(weight)

    def get_weight(self):
        return self.w
    
        
    def setupParams(self):
        temp=self.C * self.KK
        self.m=max(int(temp),1)
        self.St=np.random.randint(1,int(self.m)+1)
        for _ in range(self.St):
            self.client_model.append([]) 
            self.client_nk.append(0)
    
    def update_model_k(self,k_th_w,k_th_index,k_th_nk):
        self.tmp_count_client_updated+=1
        self.client_model[k_th_index]=k_th_w
        self.client_nk[k_th_index]=k_th_nk
        self.total_sample+=k_th_nk
        if( self.tmp_count_client_updated== len(self.client_model)):
            self.calculate_average_model()
        
    def calculate_average_model(self):
        tmp=[]
        for i in range(len(self.client_model)):
            x_frac=self.client_nk[i]/self.total_sample 
            xxx=x_frac * np.array(self.client_model[i])
            tmp+=xxx.tolist()
        print(np.array(self.w)-np.array(tmp))
        self.w=tmp

class MyFederatedServer(m.BaseManager):
    pass
MyFederatedServer.register("Federated", Federated)

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()


train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)

test_x = tf.expand_dims(test_x, 3)


val_x = train_x[:5000]
val_y = train_y[:5000]

train_x=train_x[5000:]
train_y=train_y[5000:]

_today=datetime.now()
folder_today="{0}".format("30-08")

if __name__ == "__main__":

    if not os.path.exists(folder_today):
        os.makedirs(folder_today)

    for c_index in range(len(ARRAY_C_FRACTION)):
        client_weight=Queue()
        c_fraction=ARRAY_C_FRACTION[c_index]

        fraction_folder="{0}/{1}".format(folder_today,c_fraction)
        if not os.path.exists(fraction_folder):
            os.makedirs(fraction_folder)

        manager = MyFederatedServer()
        manager.start()
        my_server = manager.Federated(c_fraction,K)
        my_server.setupParams()
        num_of_clients=my_server.getNumberOfClient()

        
            
        processes=[]
        trains_images_array=np.array_split(train_x,num_of_clients)
        trains_lables_array=np.array_split(train_y,num_of_clients)
        
        w1=np.random.rand(3*3*1*8)*0.01
        w1=np.reshape(w1,newshape=(3,3,1,8))
        w2=np.random.rand(8)*0.01
        w2=np.reshape(w2,newshape=(8,))
        w3=np.random.rand(1352*10)*0.01
        w3=np.reshape(w3,newshape=(1352,10))
        w4=np.random.rand(10)*0.01
        w4=np.reshape(w4,newshape=(10,))
        init_weight=[]
        init_weight.append(w1)
        init_weight.append(w2)
        init_weight.append(w3)
        init_weight.append(w4)
        
        my_server.set_weights(init_weight)
        
        weight_queue=Queue()

        for round_j in range(NUMBER_OF_ROUND):
            
            client_weight=my_server.get_weight()
            
            for i in range(num_of_clients):     
                
                process=my_client(weight_queue)
                process.set_weight(init_weight)
                process.set_train_x_train_y(trains_images_array[i],trains_lables_array[i])
                process.start()
                processes.append(process)
        
            for ii in range(0,num_of_clients):
                processes[ii].join()
                
            for iii in range(0,num_of_clients):
                w_client=weight_queue.get()
                print(np.array(w_client)-np.array(init_weight))
                my_server.update_model_k(w_client,iii,processes[iii].get_sample_count())
                processes[iii].terminate()

            server_weight_round_j=my_server.get_weight()
            xxx=client_weight-np.array(server_weight_round_j)
            print(xxx)
            tmp_server_weight=[]
            for kk in range(len(server_weight_round_j)):
                arr=server_weight_round_j[kk].flatten().tolist() 

                tmp_server_weight.append(arr)   

            my_server.set_weights(server_weight_round_j)

            server_model_round="{0}/round_{1}".format(fraction_folder,round_j)
            write_list(server_model_round,tmp_server_weight)

