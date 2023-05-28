import tensorflow_federated as tff
import tensorflow as tf
import numpy as np

#load mnist data
mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

#splitting data for two clients
NUM_CLIENTS = 2
NUM_EPOCHS = 1
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

def preprocess(dataset):
    def batch_format_fn(element):
        return (tf.reshape(element['pixels'],[-1,784]),tf.reshape(element['label'],[-1,1]))
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(2)

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

