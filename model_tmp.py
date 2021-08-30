from multiprocessing.context import Process
from typing import Any, ByteString
from numpy.lib.function_base import gradient
from layer import *
from inout import plot_sample, plot_learning_curve, plot_accuracy_curve, plot_histogram
import numpy as np
import time
from federated_config import *

from read_write_file import *
from inout import load_mnist, load_cifar, preprocess
dataset_name = 'mnist'
 

class Network(Process):
    def __init__(self,_learning_rate,_train_data,_regularization) :
        self.learning_rate=_learning_rate
        self.train_data=_train_data
        self.regularization=_regularization
        Process.__init__(self)  # Call the parent class initialization method
        self.layers=[]
        self.trains=Any
    def set_training_set(self,_trains):
        self.trains=_trains

    def run(self):    
        self.train(NUMBER_OF_EPOCH,self.learning_rate,self.regularization)
        
    def add_layer(self,layer):
        self.layers.append(layer)
    
    
    def build_model(self,dataset_name):
        if dataset_name == 'mnist':
           
            self.add_layer(Convolutional(name='conv1', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=8 * 6 * 6, num_classes=10))
            # total_weights=0
            # for i in range(len(self.layers)):
            #     total_weights+=len(self.layers[i].get_weights())
                    
        else:
            self.add_layer(Convolutional(name='conv1', num_filters=32, stride=1, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=32, stride=1, size=3, activation='relu'))
            self.add_layer(Pooling(name='pool1', stride=2, size=2))
            self.add_layer(Convolutional(name='conv3', num_filters=64, stride=1, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv4', num_filters=64, stride=1, size=3, activation='relu'))
            self.add_layer(Pooling(name='pool2', stride=2, size=2))
            self.add_layer(FullyConnected(name='fullyconnected', nodes1=64 * 5 * 5, nodes2=256, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=256, num_classes=10))
    
    def forward(self,image):

        for layer in self.layers:
            image=layer.forward(image)
        return image
    
    def backward(self,gradient,learning_rate):
        for layer in reversed(self.layers):
            gradient=layer.backward(gradient,learning_rate)

    def set_weights_for_layer(self,weights):
        
        start_index=0
        end_index=0

        for i in range(0,len(self.layers)):
            end_index+=len(self.layers[i].get_weights())
            _weights=weights[start_index:end_index]
            start_index=end_index
            self.layers[i].set_weights(_weights)
        
    def train(self,num_epochs,learning_rate,regularization):    
        batch_size=B
        images_train=self.trains["train_images"]
        labels_train=self.trains["train_labels"]
        train_size=images_train.shape[0]
        
        num_of_batch=1
        if train_size>batch_size:
            num_of_batch=int(train_size/batch_size)
        
        for epoch in range(0, num_epochs ):
            count =0
            print('\n--- Epoch {0} ---'.format(epoch))
        
            train_images_array=np.array_split(images_train,num_of_batch)
            train_labels_array=np.array_split(labels_train,num_of_batch)
            
            tmp_loss,num_corr=0,0
            initial_time = time.time()

            for b in range(len(train_images_array)):
                train_images=train_images_array[b]
                train_labels=train_labels_array[b]

                
                
                for i ,(image,label) in enumerate(zip(train_images,train_labels)):
                    count=count+1
                    
                    if i % batch_size == 0:
                        executed_time=time.time()-initial_time
                        initial_time= time.time()

                        accuracy = (num_corr / (count + 1)) * 100       # compute training accuracy and loss up to iteration i
                        loss = tmp_loss / (count + 1)


                        print('[Step %d] Past %d steps Time [%.fs]: Average Loss %.3f | Accuracy: %.4f' %(count + 1,batch_size,executed_time, loss , accuracy))
                        
                        
                    
                    tmp_output = self.forward(image)       # forward propagation

                    # compute (regularized) cross-entropy and update loss
                    tmp_loss += regularized_cross_entropy(self.layers, regularization, tmp_output[label])

                    if np.argmax(tmp_output) == label:                          # update accuracy
                        num_corr += 1

                    gradient = np.zeros(10)                                     # compute initial gradient
                    gradient[label] = -1 / tmp_output[label] + np.sum(
                        [2 * regularization * np.sum(np.absolute(layer.get_weights())) for layer in self.layers])

                    learning_rate = lr_schedule(learning_rate, iteration=i)     # learning rate decay

                    self.backward(gradient, learning_rate)                      # backward propagation

        
    
    def get_plot_val(self):
        return self.plot_val
    
    def evaluate(self,X,y,regularization,plot_correct,plot_missclassified,plot_feature_maps,verbose):
        loss,num_correct=0,0
        for i in range(len(X)):
            tmp_output=self.forward(X[i])
            
            loss+=regularized_cross_entropy(self.layers,regularization,tmp_output[y[i]])

            prediction =np.argmax(tmp_output)

            if prediction ==y[i]:
                num_correct +=1

                # if plot_correct:
                #     image=(X[i]*255)[0,:,:]
                #     plot_sample(image,y[i],prediction)
                #     plot_correct=1
                # else:
                #     if plot_missclassified:
                #         image=(X[i]*255)[0,:,:]
                #         plot_sample(image,y[i],prediction)
                #         plot_missclassified=1
            
        test_size=len(X)
        accuracy =(num_correct/test_size)*100
        loss = loss /test_size

        # if verbose:
        #     print('Test Loss: %02.3f' % loss)
        #     print('Test Accuracy: %02.3f' % accuracy)
        return loss,accuracy

    def get_layer_weights(self):
        weights=[]
        for i in range(0,len(self.layers)):
            weights+=self.layers[i].get_weights().tolist()
        return weights
    
    def get_layer_weights_len(self):
        weight_len=0
        for i in range(0,len(self.layers)):
            weight=self.layers[i].get_weights().tolist()
            weight_len+=len(weight)
        return weight_len

    def get_sample_count(self):
        return len(self.trains["train_labels"])
        
        
    
        
        