from multiprocessing.context import Process
from typing import ByteString
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
    def __init__(self,_thread_index,
    _data_set,
        _learning_rate,_validate,_regularization,
        _plot_weights,_verbose) :
        Process.__init__(self)  # Call the parent class initialization method
        self.layers=[]
        self.thread_index=_thread_index
        self.data_set=_data_set
        # dataset = load_mnist() if dataset_name == 'mnist' else load_cifar()
        # dataset = preprocess(dataset)
        random_sample_splited=np.random.randint(1,10,dtype=int)
        
        
        train_images=self.data_set['train_images']
        train_lables=self.data_set['train_labels']


        trains_images_array=np.array_split(train_images,random_sample_splited)
        trains_lables_array=np.array_split(train_lables,random_sample_splited)
        trains={'train_images':trains_images_array[0], 'train_labels':trains_lables_array[0]}

        self.trains=trains

        
        
        self.learning_rate=_learning_rate
        self.validate=_validate
        self.regularization=_regularization
        self.plot_weights=_plot_weights
        self.verbose=_verbose
        
        
        self.plot_val=[]
    
    def __init__(self,
        _learning_rate,_validate,_regularization,
        _plot_weights,_verbose) :
        Process.__init__(self)  # Call the parent class initialization method
        self.layers=[]
        
        self.learning_rate=_learning_rate
        self.validate=_validate
        self.regularization=_regularization
        self.plot_weights=_plot_weights
        self.verbose=_verbose
        
        
        self.plot_val=[]
    


    def run(self):    
        #self.build_model("mnist")
        self.train(
            self.trains,
            1,
            self.learning_rate,
            self.validate,
            self.regularization,
            self.plot_weights,
            self.verbose)
        
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
    def forward(self,image,plot_feature_maps):
        global history
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
        
    def train(self,
                dataset,num_epochs,learning_rate,validate,regularization,plot_weights,verbose):    
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
                
            for b in range(len(train_images_array)):
                train_images=train_images_array[b]
                train_labels=train_labels_array[b]

                loss = 0
                num_correct = 0
                initial_time = time.time()
                for i ,(image,label) in enumerate(zip(train_images,train_labels)):
                    count=count+1
                   # print(i)
                    if i % B == B-1:
                        #print("{0}/{1} {2}".format(count,train_size,100*(count/train_size)))
                        executed_time=time.time()-initial_time
                        initial_time= time.time()
                        
                        #print("[Step {0}] Past 100 steps Time [{1}s]: Average Loss {2} | Accuracy: {3}".format(count + 1,executed_time, loss / B),float(num_correct)/B*100)
                        print('[Step %d] Past 100 steps Time [%.fs]: Average Loss %.3f | Accuracy: %d%%' %(count + 1,executed_time, loss / batch_size, num_correct/batch_size*100))
                        loss = 0
                        num_correct = 0

                        
                    
                   
                    tmp_output = self.forward(image, plot_feature_maps=0)       # forward propagation
                   
                    l = -np.log(tmp_output[label])
                    acc = 1 if np.argmax(tmp_output) == label else 0
                    # compute (regularized) cross-entropy and update loss
                    #tmp_loss += regularized_cross_entropy(self.layers, regularization, tmp_output[label])
                    
                    loss += l
                    num_correct += acc



                    gradient = np.zeros(10)                                     # compute initial gradient
                    gradient[label] = -1 / tmp_output[label] + np.sum(
                        [2 * regularization * np.sum(np.absolute(layer.get_weights())) for layer in self.layers])

                    learning_rate = lr_schedule(learning_rate, iteration=i) 

                    self.backward(gradient, learning_rate)                      # backward propagation
                    
        
    
    def get_plot_val(self):
        return self.plot_val
    
    def evaluate(self,X,y,regularization,plot_correct,plot_missclassified,plot_feature_maps,verbose):
        loss,num_correct=0,0
        for i in range(len(X)):
            tmp_output=self.forward(X[i],plot_feature_maps)
            
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
        
        
    
        
        