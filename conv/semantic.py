#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import dnn


def build_model(input_dim, output_dim, batch_size=None):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 1, 1, input_dim),
    )
    l_conv1 = dnn.Conv2DDNNLayer(
        l_in,
        num_filters=32,
        filter_size=(1, 11),
        strides=(1, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool1 = dnn.MaxPool2DDNNLayer(
        l_conv1,
        ds=(1, 2),
    )
    l_conv2 = dnn.Conv2DDNNLayer(
        l_pool1,
        num_filters=64,
        filter_size=(1, 5),
        strides=(1, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool2 = dnn.MaxPool2DDNNLayer(
        l_conv2,
        ds=(1, 2),
    )
    l_conv3 = dnn.Conv2DDNNLayer(
        l_pool2,
        num_filters=128,
        filter_size=(1, 3),
        strides=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool3 = dnn.MaxPool2DDNNLayer(
        l_conv3,
        ds=(1, 2),
    )
    l_conv4 = dnn.Conv2DDNNLayer(
        l_pool3,
        num_filters=256,
        filter_size=(1, 3),
        strides=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool4 = dnn.MaxPool2DDNNLayer(
        l_conv4,
        ds=(1, 2),
    )
    l_conv5 = dnn.Conv2DDNNLayer(
        l_pool4,
        num_filters=512,
        filter_size=(1, 3),
        strides=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool5 = dnn.MaxPool2DDNNLayer(
        l_conv5,
        ds=(1, 2),
    )
    l_fc1 = lasagne.layers.DenseLayer(
        l_pool5,
        num_units=500,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    #l_hidden1_dropout = lasagne.layers.DropoutLayer(
    #    l_hidden1,
    #    p=0.5,
    #)
    l_out = lasagne.layers.DenseLayer(
        l_fc1,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    return l_out
 
