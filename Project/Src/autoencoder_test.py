import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high,\
        dtype=tf.float32)


class VariationalAutoencoder(object):
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,\
            learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None,
        network_architecture["n_input"]])
    
        #create autoencoder network
        self._create_network()

        #define loss function
        self._create_loss_optimizer()

        #launch session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

       
def _create_network(self):
    #initialize network and weights
    network_weights = self._initialize_weights(**self.network_architecture)

    #get variance and mean of gaussian distribution in latent space
    self.z_mean, sel.z_log_sigma_sq = \
        self._recognition_network(network_weights["weights_recog"],
                                  network_weights["biases_recog"])


    #draw sample z from gaussian distribution
    n_z = self.network_architecture["n_z"]
    eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)

    #z = mu + sigma * epsilon
    self.z = tf.add(self.z_mean,\
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    self.x_reconstr_mean = \
        self._generator_network(network_weights["weights_gener"],
                                network_weights["biases_gener"])




