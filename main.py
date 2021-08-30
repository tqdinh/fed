from model_tmp import Network
from inout import load_mnist, load_cifar, preprocess
from read_write_file import *

if __name__ == '__main__':

    '''
        Hyper parameters
        
            - dataset_name              choose between 'mnist' and 'cifar'
            - num_epochs                number of epochs
            - learning_rate             learning rate
            - validate                  0 -> no validation, 1 -> validation
            - regularization            regularization term (i.e., lambda)
            - verbose                   > 0 --> verbosity
            - plot_weights              > 0 --> plot weights distribution
            - plot_correct              > 0 --> plot correct predicted digits from test set
            - plot_missclassified       > 0 --> plot missclassified digits from test set
            - plot_feature_maps         > 0 --> plot feature maps of predicted digits from test set
    '''

    dataset_name = 'mnist'
    num_epochs = 1
    learning_rate = 0.01
    validate = 0
    regularization = 0
    verbose = 1
    plot_weights = 0
    plot_correct = 0
    plot_missclassified = 0
    plot_feature_maps = 0

    print('\n--- Loading ' + dataset_name + ' dataset ---')                 # load dataset
    dataset = load_mnist() if dataset_name is 'mnist' else load_cifar()

    print('\n--- Processing the dataset ---')                               # pre process dataset
    dataset = preprocess(dataset)

    print('\n--- Building the model ---')                                   # build model
    model = Network(learning_rate,validate,regularization)
    model.build_model(dataset_name)

    model.set_training_set(dataset)
    print('\n--- Training the model ---')                                   # train model
    # model.train(
    #     num_epochs,
    #     learning_rate,
    #     regularization,
    # )
    #www=model.get_layer_weights()

    round_find="{0}/client{1}".format("TEST","x")
    # write_list(round_find,www)
    www=read_list(round_find)

    model.set_weights_for_layer(www)
    print('\n--- Testing the model ---')                                    # test model
    val_loss,val_accuracy=model.evaluate(
        dataset['test_images'],
        dataset['test_labels'],
        regularization,
        plot_correct,
        plot_missclassified,
        plot_feature_maps,
        verbose
    )

    print("----------------------LOSS----------------")                            
    print('Valid Loss: %02.3f' % val_loss)
    print('valid Accuracy: %02.3f' % val_accuracy)