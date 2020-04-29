import numpy as np
import pandas as pd
import tensorflow as tf

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool_2x2(x, ksize=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')

def dropout_layer(x, keep_prob):
    return tf.nn.dropout(x, keep_prob=keep_prob)

def conv_layer(input, input_channels, filters, filters_size):
    weights = tf.Variable(tf.truncated_normal(shape=[filters_size, filters_size, input_channels, filters], mean=0, stddev=0.05))
    biases = tf.Variable(tf.zeros([filters]))

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    layer = tf.nn.relu(layer)

    return layer

def pool(layer, ksize, strides, padding='SAME'):
    return tf.nn.max_pool(layer, ksize=ksize, strides=strides, padding=padding)

def flatten_layer(layer):
    shape = layer.get_shape()
    features = shape[1:4].num_elements()
    
    layer = tf.reshape(layer, [-1, features])

    return layer

def fc_layer(input, inputs, outputs, relu=True, is_linear=False):

    layer = None

    if is_linear:
        weights = tf.Variable(tf.truncated_normal(shape=[inputs.get_shape()[-1], outputs], mean=0, stddev=0.05))
        biases = tf.Variable(tf.zeros([outputs]))

        layer = tf.matmul(inputs, weights)

    else:
        weights = tf.Variable(tf.truncated_normal(shape=[inputs, outputs], mean=0, stddev=0.05))
        biases = tf.Variable(tf.zeros([outputs]))

        layer = tf.matmul(input, weights)
    
    layer += biases

    if relu:
        layer = tf.nn.relu(layer)

    return layer
    