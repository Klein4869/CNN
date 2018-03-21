import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
from PIL import Image
from scipy import ndimage
import gzip
import numpy as np

imagesize = 28
pixel_depth = 255


def extract_data(filename, num_images):
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(imagesize * imagesize * num_images * 1)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (pixel_depth - 2.0)) / pixel_depth
        data = data.reshape(num_images, imagesize, imagesize, 1)
        return data


def extract_labels(filename, num_images):
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        labels = labels.reshape(num_images,1)
        return labels


def random_mini_batches(X_train, Y_train, minibatch_size, seed):
    x_batch, y_batch = tf.train.shuffle_batch([X_train, Y_train], minibatch_size, capacity=300, min_after_dequeue=100)
    minibatches = {"x_batch": x_batch,
                   "y_batch": y_batch}
    return minibatches


def create_placeholder(n_H, n_W, n_C, n_Y):
    x = tf.placeholder(tf.float32, [None, n_H, n_W, n_C])
    y = tf.placeholder(tf.float32, [None, n_Y])
    return x, y


def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [3, 3, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [3, 3, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)

    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)

    Z3 = tf.nn.conv2d(A2, W3, strides=[1, 1, 1, 1], padding='SAME')

    P2 = tf.nn.max_pool(Z3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    P2 = tf.contrib.layers.flatten(P2)

    Z4 = tf.contrib.layers.fully_connected(P2, 10)

    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.005, num_epochs=100, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_Y = 10
    costs = []

    X, Y = create_placeholder(n_H0, n_W0, n_C0, n_Y)

    parameters = initialize_parameters()

    Z4 = forward_propagation(X, parameters)

    cost = compute_cost(Z4, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(0,num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed += 1
            #minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for step in range():
                print(minibatch)
                minibatch_x = minibatch['x_batch']
                minibatch_y = minibatch['y_batch']

                sess.run(optimizer, feed_dict={X: minibatch_x, Y: minibatch_y})
                temp_cost = sess.run(cost, feed_dict={X: minibatch_x, Y: minibatch_y})

                minibatch_cost += temp_cost / minibatches

        if epoch % 1 and print_cost == True:
            print_cost("Cost after epoch %i: %f" % (epoch, minibatch_cost))

train_data_f = "/Users/haidongtang/Desktop/CNN EXAMPLES/mnist1/train-images-idx3-ubyte.gz"
train_labels_f = "/Users/haidongtang/Desktop/CNN EXAMPLES/mnist1/train-labels-idx1-ubyte.gz"
test_data_f = "/Users/haidongtang/Desktop/CNN EXAMPLES/mnist1/t10k-images-idx3-ubyte.gz"
test_labels_f = "/Users/haidongtang/Desktop/CNN EXAMPLES/mnist1/t10k-labels-idx1-ubyte.gz"

train_data = extract_data(train_data_f, 60000)
train_labels = extract_labels(train_labels_f, 60000)
test_data = extract_data(test_data_f, 10000)
test_labels = extract_labels(test_labels_f, 10000)

print(train_labels.shape)

model(train_data, train_labels, test_data, test_labels)