"""
    Custom layers for WaveNet
"""
from __future__ import division
import numpy as np
import tensorflow as tf

def create_variable(name, shape, seed=None):
    """ Create variable with Xavier initialization """
    init = tf.contrib.layers.xavier_initializer(seed=seed)
    return tf.get_variable(name=name, shape=shape, initializer=init)

def create_bias_variable(name, shape):
    """ Create variable with zeros initialization """
    init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=init)

def time_to_batch(inputs, dilation):
    """ If necessary zero-pads inputs and reshape by dilation """
    with tf.variable_scope('time_to_batch'):
        _, width, num_channels = inputs.get_shape().as_list()

        width_pad = int(dilation * np.ceil((width + dilation) * 1.0 / dilation))
        pad_left = width_pad - width

        shape = (int(width_pad / dilation), -1, num_channels)
        padded = tf.pad(inputs, [[0, 0], [pad_left, 0], [0, 0]])
        transposed = tf.transpose(padded, (1, 0, 2))
        reshaped = tf.reshape(transposed, shape)
        outputs = tf.transpose(reshaped, (1, 0, 2))
        return outputs

def batch_to_time(inputs, dilation, crop_left=0):
    """ Reshape to 1d signal, and remove excess zero-padding """
    with tf.variable_scope('batch_to_time'):
        shape = tf.shape(inputs)
        batch_size = shape[0] / dilation
        width = shape[1]
        
        out_width = tf.to_int32(width * dilation)
        _, _, num_channels = inputs.get_shape().as_list()
        
        new_shape = (out_width, -1, num_channels) # missing dim: batch_size
        transposed = tf.transpose(inputs, (1, 0, 2))    
        reshaped = tf.reshape(transposed, new_shape)
        outputs = tf.transpose(reshaped, (1, 0, 2))
        cropped = tf.slice(outputs, [0, crop_left, 0], [-1, -1, -1])
        return cropped

def conv1d(inputs, out_channels, filter_width=2, stride=1, padding='VALID', 
        activation=tf.nn.relu, seed=None, bias=True, name='conv1d'):
    """ Normal 1D convolution operator """ 
    with tf.variable_scope(name):
        in_channels = inputs.get_shape().as_list()[-1]

        W = create_variable('W', (filter_width, in_channels, out_channels), seed)
        outputs = tf.nn.conv1d(inputs, W, stride=stride, padding=padding)

        if bias:
            b = create_bias_variable('bias', (out_channels, ))
            outputs += tf.expand_dims(tf.expand_dims(b, 0), 0)

        if activation:
            outputs = activation(outputs)

        return outputs

def dilated_conv(inputs, out_channels, filter_width=2, dilation=1, stride=1, 
        padding='VALID', name='dilated_conv', activation=tf.nn.relu, seed=None):
    """ Warpper for 1D convolution to include dilation """
    with tf.variable_scope(name):
        width = inputs.get_shape().as_list()[1]

        inputs_ = time_to_batch(inputs, dilation)
        outputs_ = conv1d(inputs_, out_channels, filter_width, stride, padding, activation, seed)

        out_width = outputs_.get_shape().as_list()[1] * dilation
        diff = out_width - width
        outputs = batch_to_time(outputs_, dilation, crop_left=diff)

        # Add additional shape information.
        tensor_shape = [tf.Dimension(None), tf.Dimension(width), tf.Dimension(out_channels)]
        outputs.set_shape(tf.TensorShape(tensor_shape))

        return outputs
