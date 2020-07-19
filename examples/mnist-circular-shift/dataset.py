import tensorflow as tf


def preprocess(x):
    """Reshape images into row vectors and scale entries to be between zero and
    one.

    """
    return x.reshape((-1, 784)) / 255.

def mnist():
    """Load MNIST dataset as training and test splits, preprocess images, and
    return as the training dataset and the test dataset.

    """
    (trainset, testset) = tf.keras.datasets.mnist.load_data()
    x_train = preprocess(trainset[0])
    y_train = trainset[1]
    x_test = preprocess(testset[0])
    y_test = testset[1]
    return (x_train, y_train), (x_test, y_test)
