import numpy as np
import multiprocessing.managers as m
from numpy.lib.function_base import select

from model_tmp import Network
from multiprocessing import shared_memory, Process, Manager,Queue
from multiprocessing import cpu_count, current_process
from datetime import datetime
from federated_config import *
from read_write_file import *

import os
from datetime import datetime
import shutil
from inout import load_mnist, load_cifar, preprocess




learning_rate = 10e-4
validate = 0
regularization = 0
verbose = 1
plot_weights = 0
plot_correct = 0
plot_missclassified = 0
plot_feature_maps = 0

dataset = load_mnist()

dataset = preprocess(dataset)


validation_images=dataset['validation_images']
validation_labels=dataset['validation_labels']


class ClientInfo:
    def __init__(self,_nk,_w):
        self.w=_w
        self.nk=_nk # number of sample

class FederatedLearning:
   
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

    def refresh(self):
        self.tmp_count_client_updated=0
        self.total_sample=0
        self.w=[]
        self.client_model=[]
        self.client_nk=[]
        self.m =0#max(C*K,1)
        self.St =0 #random set of client 
    def getNumberOfClient(self):
        return self.St
   
    def setupParams(self):
        temp=self.C * self.KK
        self.m=max(int(temp),1)
        self.St=np.random.randint(1,int(self.m)+1)
        for _ in range(self.St):
            self.client_model.append([]) 
            self.client_nk.append(0)
   
    def getw(self):
        return self.w
    def setw(self,_w):
        self.w=_w

    def update_model_k(self,k_th_w,k_th_index,k_th_nk):
        self.tmp_count_client_updated+=1
        self.client_model[k_th_index]=k_th_w
        self.client_nk[k_th_index]=k_th_nk
        self.total_sample+=k_th_nk
        if( self.tmp_count_client_updated== len(self.client_model)):
            self.calculate_average_model()
        
    def get_w(self):
        return self.w

    def calculate_average_model(self):
        tmp=[]
        for i in range(len(self.client_model)):
            x_frac=self.client_nk[i]/self.total_sample 
            xxx=x_frac* np.array(self.client_model[i])
            tmp+=xxx.tolist()
        self.w=tmp

class MyFederatedServer(m.BaseManager):
    pass
MyFederatedServer.register("FederatedLearning", FederatedLearning)


if __name__ == "__main__":


    dataset = load_mnist()

    dataset = preprocess(dataset)

    train_images=dataset['train_images']
    train_lables=dataset['train_labels']
    validation_images=dataset['validation_images']
    validation_labels=dataset['validation_labels']

    _today=datetime.now()
    folder_today="{0}".format(_today)
        
    if not os.path.exists(folder_today):
        os.makedirs(folder_today)
        
        original = "plot_my_graph.py"
        target = "{0}/{1}".format(folder_today,original)
        shutil.copyfile(original, target)

        original1 = "read_write_file.py"
        target1 = "{0}/{1}".format(folder_today,original1)
        shutil.copyfile(original1, target1)

    for c_index in range(len(ARRAY_C_FRACTION)):
        



        c_fraction=ARRAY_C_FRACTION[c_index]
        
        print("trainning with fraction",c_fraction)


        

        manager = MyFederatedServer()
        manager.start()
        my_server = manager.FederatedLearning(c_fraction,K)
        my_server.refresh()
        
        list_accuracy=[]
        list_loss=[]

        for j in range(NUMBER_OF_ROUND):
            my_server.setupParams()
            num_of_clients=my_server.getNumberOfClient()
            processes=[]
            folder_name="{0}".format(folder_today)

            my_queue=Queue()
            my_queue_accuracy=Queue()

            list_of_loss_thread=[]
            list_of_accuracy_thread=[]

            trains_images_array=np.array_split(train_images,num_of_clients)
            trains_lables_array=np.array_split(train_lables,num_of_clients)
            trains=[]
            for i in range(len(trains_images_array)):
                trains.append({'train_images':trains_images_array[i], 'train_labels':trains_lables_array[i]})
        

            for j in range(num_of_clients):
                list_of_loss_thread.append([])
                list_of_accuracy_thread.append([])

            for i in range(num_of_clients):
                process=Network(learning_rate,validate,regularization)
                process.build_model("mnist")
                process.set_training_set(trains[i])
                if(0==len(my_server.getw())):
                    len_w=process.get_layer_weights_len()
                    w0=np.random.rand(len_w)*0.1
                    my_server.setw(w0)

                process.start()
                processes.append(process)
        
            for ii in range(0,num_of_clients):
                processes[ii].join()
            
            for iii in range(0,num_of_clients):
                w_client=processes[iii].get_layer_weights()
                round_find="{0}/client{1}".format("TEST",iii)
                write_list(round_find,w_client)
                my_server.update_model_k(w_client,iii,processes[iii].get_sample_count())

            www=my_server.get_w()
            round_find="{0}/fraction_{1}".format(folder_today,c_fraction)
            write_list(round_find,www)


            valid_model = Network(learning_rate,validate,regularization)

            valid_model.build_model("mnist")
            valid_model.set_weights_for_layer(www)

            indices=np.random.permutation(dataset['validation_images'].shape[0])
            val_loss,val_accuracy=valid_model.evaluate(validation_images[indices, :],
                                    validation_labels[indices],
                                    regularization,
                                    plot_correct=0,
                                    plot_missclassified=0,
                                    plot_feature_maps=0,
                                    verbose=0)
            print("----------------------LOSS----------------")                            
            print('Valid Loss: %02.3f' % val_loss)
            print('valid Accuracy: %02.3f' % val_accuracy)
            list_accuracy.append(val_accuracy)
            list_loss.append(val_loss)
        
        accuracy_file="{0}/accuracy_{1}".format(folder_today,c_fraction)
        write_list(accuracy_file,val_accuracy)

        loss_file="{0}/loss_{1}".format(folder_today,c_fraction)
        write_list(loss_file,list_loss)

