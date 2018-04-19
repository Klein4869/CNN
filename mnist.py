import input_data_mnist
import numpy as np

mnist = input_data_mnist.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf
from tensorflow.python.framework import ops


def create_placeholder(n_H, n_W, n_C, n_Y):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, n_Y])
    return x, y


def initial_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable('W1', [3, 3, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', [3, 3, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable('W3', [3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {'W1': W1,
                  'W2': W2,
                  'W3': W3}

    return parameters


def forward_propagation(X, parameters):
    X = tf.reshape(X, [-1, 28, 28, 1])
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

    return Z4


def compute_cost(Z4, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z4, labels=Y))

    return cost


def model(learning_rate=0.01, num_training=20000, minibatch_size=100):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = (60000, 28, 28, 1)
    n_Y = 10

    X, Y = create_placeholder(28, 28, 1, n_Y)

    parameters = initial_parameters()

    Z4 = forward_propagation(X, parameters=parameters)

    cost = compute_cost(Z4, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(0, num_training):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed += 1
            minibatch = mnist.train.next_batch(minibatch_size)
            # for minibatch in minibatches:
            #     sess.run(optimizer, feed_dict={X: minibatch[0], Y: minibatch[1]})
            #     temp_cost = sess.run(cost, feed_dict={X: minibatch[0], Y: minibatch[1]})
            #
            #     minibatch_cost += temp_cost
            sess.run(optimizer, feed_dict={X: minibatch[0], Y: minibatch[1]})
            minibatch_cost = sess.run(cost, feed_dict={X: minibatch[0], Y: minibatch[1]})
            if epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                predict_op = tf.argmax(Z4, 1)
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("The Train accuracy is: %f" % accuracy.eval(feed_dict={X: minibatch[0], Y: minibatch[1]}))
                print(
                    "The Test accuracy is: %f" % accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
                print()


model()
