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
    # def __init__(self,_thread_index,
    #     _num_epochs,_learning_rate,_validate,_regularization,
    #     _plot_weights,_verbose,_list_of_loss_thread,_queue) :
    #     Process.__init__(self)  # Call the parent class initialization method
    #     self.layers=[]
        
    #     self.thread_index=_thread_index
        
    #     dataset = load_mnist() if dataset_name == 'mnist' else load_cifar()
    #     dataset = preprocess(dataset)
    #     random_sample_splited=np.random.randint(5,10,dtype=int)
        
    #     train_images=dataset['train_images']
    #     train_lables=dataset['train_labels']


    #     trains_images_array=np.array_split(train_images,random_sample_splited)
    #     trains_lables_array=np.array_split(train_lables,random_sample_splited)
    #     trains={'train_images':trains_images_array[0], 'train_labels':trains_lables_array[0]}

    #     self.trains=trains
    #     self.num_epochs=_num_epochs
    #     self.learning_rate=_learning_rate
    #     self.validate=_validate
    #     self.regularization=_regularization
    #     self.plot_weights=_plot_weights
    #     self.verbose=_verbose
    #     self.list_of_loss_thread=_list_of_loss_thread
    #     self.plot_val=[]
    #     self.queue_loss=_queue
    
    def __init__(self,_thread_index,
        _num_epochs,_learning_rate,_validate,_regularization,
        _plot_weights,_verbose,_list_of_loss_thread,_queue_loss,_queue_accuracy,_my_folder_to_save) :
        Process.__init__(self)  # Call the parent class initialization method
        self.layers=[]
        self.thread_index=_thread_index


        self.thread_index=_thread_index
        
        dataset = load_mnist() if dataset_name == 'mnist' else load_cifar()
        dataset = preprocess(dataset)
        #random_sample_splited=np.random.randint(1,10,dtype=int)
        random_sample_splited=10000
        
        train_images=dataset['train_images']
        train_lables=dataset['train_labels']


        trains_images_array=np.array_split(train_images,random_sample_splited)
        trains_lables_array=np.array_split(train_lables,random_sample_splited)
        trains={'train_images':trains_images_array[0], 'train_labels':trains_lables_array[0]}

        self.trains=trains

        
        self.num_epochs=_num_epochs
        self.learning_rate=_learning_rate
        self.validate=_validate
        self.regularization=_regularization
        self.plot_weights=_plot_weights
        self.verbose=_verbose
        self.list_of_loss_thread=_list_of_loss_thread
        
        self.plot_val=[]
        self.queue_loss=_queue_loss
        self._queue_accuracy=_queue_accuracy
        self.my_folder_to_save=_my_folder_to_save
    
    
    def run(self):    
        #self.build_model("mnist")
        self.train(
            self.trains,
            self.num_epochs,
            self.learning_rate,
            self.validate,
            self.regularization,
            self.plot_weights,
            self.verbose)
        
    def add_layer(self,layer):
        self.layers.append(layer)
    
    
    def build_model(self,dataset_name):
        if dataset_name == 'mnist':
            # self.add_layer(Convolutional(name='conv1', num_filters=8, stride=1, size=3, activation='xxrelu'))
            # self.add_layer(Pooling(name='pool0', stride=2, size=2))
            # self.add_layer(Dense(name='dense', nodes=8 * 13 * 13, num_classes=10))

            self.add_layer(Convolutional(name='conv1', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=8 * 6 * 6, num_classes=10))

            # self.add_layer(Convolutional(name='conv1', num_filters=6, stride=1, size=3, activation='relu'))
            # self.add_layer(Pooling(name='pool1', stride=2, size=2))
            # self.add_layer(Convolutional(name='conv2', num_filters=16, stride=1, size=3, activation='relu'))
            # self.add_layer(Pooling(name='pool2', stride=2, size=2))
            # self.add_layer(Dense(name='dense', nodes=400, num_classes=10))
            total_weights=0
            for i in range(len(self.layers)):
                w_layer_i=len(self.layers[i].get_weights())
                #print("layer w_ {0}".format(w_layer_i))
                total_weights+=w_layer_i
                
            rand=np.random.rand(total_weights)*0.1
            
            

                    
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
#            print('\nbackward class {0}'.format(layer.name))
            gradient=layer.backward(gradient,learning_rate)
            weight=layer.get_weights()
            #print('\nname={0}   weight {1}'.format(layer.name,weight.shape))
#           print('\nbackward  class gradient {0}'.format(gradient))
    #    print('\nbackward gradient {0}'.format(gradient))


    
    def train(self,
                dataset,num_epochs,learning_rate,validate,regularization,plot_weights,verbose):    
        
        plotting=[]
        plotting_accuracy=[]
        batch_size=B
        local_accuracy=[]
        local_loss=[]
        t1=time.time()
        
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
       
        images_train=self.trains["train_images"]
        labels_train=self.trains["train_labels"]
        train_size=images_train.shape[0]
        
        num_of_batch=1
        if train_size>batch_size:
            num_of_batch=int(train_size/batch_size)
        
        for epoch in range(0, num_epochs ):
            count =0
            print('\n--- Epoch {0} ---'.format(epoch))
            #print(list_time_stamp) 
          #  permutation = np.random.permutation(len(dataset["train_images"]))
            
            # train_images = train_images[permutation]
            # train_labels = train_labels[permutation]

            train_images_array=np.array_split(images_train,num_of_batch)
            train_labels_array=np.array_split(labels_train,num_of_batch)
            
            
            
            for b in range(len(train_images_array)):
                train_images=train_images_array[b]
                train_labels=train_labels_array[b]
                
                #train_image_len=len(train_images)
                # permutation = np.random.permutation(int(train_image_len))
                # train_images = train_images[permutation]
                # train_labels = train_labels[permutation]

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
                        local_accuracy.append(num_correct)    
                        #print("[Step {0}] Past 100 steps Time [{1}s]: Average Loss {2} | Accuracy: {3}".format(count + 1,executed_time, loss / B),float(num_correct)/B*100)
                        print('[Step %d] Past 100 steps Time [%.fs]: Average Loss %.3f | Accuracy: %d%%' %(count + 1,executed_time, loss / batch_size, num_correct/batch_size*100))
                        loss = 0
                        num_correct = 0

                        history['loss'].append(loss)                # update history
                        history['accuracy'].append(float(num_correct)/B)
            
                    #image=np.pad(image, ((0,0),(2,2),(2,2)),constant_values=0)
                   
                    tmp_output = self.forward(image, plot_feature_maps=0)       # forward propagation
                   
                    l = -np.log(tmp_output[label])
                    acc = 1 if np.argmax(tmp_output) == label else 0
                    # compute (regularized) cross-entropy and update loss
                    #tmp_loss += regularized_cross_entropy(self.layers, regularization, tmp_output[label])
                    local_loss.append(l)
                    loss += l
                    num_correct += acc



                    gradient = np.zeros(10)                                     # compute initial gradient
                    gradient[label] = -1 / tmp_output[label] + np.sum(
                        [2 * regularization * np.sum(np.absolute(layer.get_weights())) for layer in self.layers])

                    learning_rate = lr_schedule(learning_rate, iteration=i) 

                    t4=time.time()
                    self.backward(gradient, learning_rate)                      # backward propagation
                    
    
            mean_loss=np.mean(local_loss)
            mean_accuracy=np.mean(local_accuracy)
            print("----   epoch:{0}  loss:{1}".format(epoch,mean_loss))
            plotting.append(mean_loss)
            plotting_accuracy.append(mean_accuracy)
            
            #plotting_info["loss_vals"][type_n_thread][epoch].append(loss)

        # file_accuracy_name="{0}/accacy_process_{1}".format(self.my_folder_to_save,self.thread_index)
        # file_loss_name="{0}/loss_process_{1}".format(self.my_folder_to_save,self.thread_index)
        
        # write_list(file_accuracy_name,plotting_accuracy)
        # write_list(file_loss_name,plotting)
          
          
        self.queue_loss.put(plotting)
        self._queue_accuracy.put(plotting_accuracy)
        return plotting
    
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
            www=self.layers[i].get_weights()
            print(type(www))
            weights=weights+www.tolist()
        return weights
    
    def get_layer_weights_len(self):
        weights=0
        for i in range(0,len(self.layers)):
            www=self.layers[i].get_weights()
            weights+=len(www)
        return weights

    def get_sample_count(self):
        return len(self.trains["train_labels"])
        
        
    def update_new_chunk(self,chunk_model,index):
        if len(self.chunk_weights ) > index:
            self.chunk_weights[index].update_chunk_with_new_model(chunk_model)
    
        
        