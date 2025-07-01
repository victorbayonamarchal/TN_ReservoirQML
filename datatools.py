from tensorflow.keras.datasets import mnist

import warnings
warnings.filterwarnings("ignore")

def get_mnist_data(n_train, n_test):
    """Extracts the first n_train, n_test samples from MNIST dataset training and test data sets, respectively
    
    -- Input:
    n_train: number of train images to be returned
    n_test: number of test images to be returned

    -- Output:
    x_train_vectors: reshaped train images as vectors
    y_train: corresponding train labels
    x_test_vectors: reshaped test images as vectors
    y_test: corresponding test labels
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    ## 

 
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    x_test = x_test[:n_test]
    y_test = y_test[:n_test]

    ##
    x_train_vectors = x_train.reshape(-1, 28*28, )
    x_test_vectors = x_test.reshape(-1, 28*28, )

    return x_train_vectors, y_train, x_test_vectors, y_test
