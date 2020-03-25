"""TensorFlow module function"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

def mlp_model(inputs, num_outputs, scope, reuse=False, num_units=64, num_layers=3):
    """simple multi-layer percetron model"""
    with tf.variable_scope(scope, reuse=reuse):
        x = inputs
        for _ in range(num_layers-1):
            x = layers.fully_connected(x, num_outputs=num_units, activation_fn=tf.nn.relu)
        x = layers.fully_connected(x, num_outputs=num_outputs, activation_fn=None)
        return x

def mlp_model_Q(inputs, num_outputs, scope, reuse=False, num_units=64, num_layers=5):
    """simple multi-layer percetron model"""
    with tf.variable_scope(scope, reuse=reuse):
        x = inputs
        for _ in range(num_layers - 1):
            x = layers.fully_connected(x, num_outputs=num_units, activation_fn=tf.nn.relu)
        x = layers.fully_connected(x, num_outputs=num_outputs, activation_fn=None)
        return x

def aggregative_model(inputs, num_outputs, scope, reuse=False, num_units=64, num_layers=2):
    """aggregative input-size-variable model"""
    """allowing the second dimension of the inputs to be None"""
    with tf.variable_scope(scope, reuse=reuse):
        x = inputs
        for _ in range(num_layers-1):
            x = layers.fully_connected(x, num_outputs=num_units, activation_fn=tf.nn.relu)
        x = layers.fully_connected(x, num_outputs=num_outputs, activation_fn=None)
        x = tf.reduce_sum(x, axis=1)
        return x

def aggregative_cnn_model(inputs, num_filters, kernel_size, aggregated_kernel_size, scope, reuse, num_layers=2):
    """1 \times 1 cnn model"""
    with tf.variable_scope(scope, reuse=reuse) as sc:
        x = inputs
        for _ in range(num_layers-1):
            x = layers.conv2d(inputs, num_filters, kernel_size)
        x = layers.conv2d(inputs, num_filters, kernel_size, activation_fn=tf.identity)
        x = layers.max_pool2d(x, aggregated_kernel_size, stride=aggregated_kernel_size)
        return x