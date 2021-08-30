from read_write_file import *
from model import Network
from inout import load_mnist, load_cifar, preprocess
folder_name="2021-08-27 16:51:15.272365"
file_name_0="{0}/fraction_{1}".format(folder_name,0.0)
file_name_1="{0}/fraction_{1}".format(folder_name,0.1)
file_name_2="{0}/fraction_{1}".format(folder_name,0.2)
file_name_3="{0}/fraction_{1}".format(folder_name,0.5)
file_name_4="{0}/fraction_{1}".format(folder_name,1.0)
list_of_weights=[file_name_0,file_name_1,file_name_2,file_name_3,file_name_4]


learning_rate = 10e-4
validate = 0
regularization = 0
verbose = 1
plot_weights = 0
plot_correct = 0
plot_missclassified = 0
plot_feature_maps = 0

weight=read_list("2021-08-27 16:51:15.272365/client0")
    
dataset = load_mnist()

dataset = preprocess(dataset)

train_images=dataset['train_images']
train_lables=dataset['train_labels']
validation_images=dataset['validation_images']
validation_labels=dataset['validation_labels']


valid_model=Network(
            learning_rate,validate,regularization,plot_weights,
            verbose)
            

valid_model.build_model("mnist")
valid_model.set_weights_for_layer(weight)

    

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

