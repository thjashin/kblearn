#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import lasagne


def build_model(input_dim, output_dim, batch_size=None):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 1, input_dim),
    )
#     l_conv1 = lasagne.layers.Conv1DLayer(
#         l_in,
#         num_filters=32,
#         filter_length=50,
#         stride=50,
#         border_mode='valid',
#         nonlinearity=lasagne.nonlinearities.rectify,
#         W=lasagne.init.GlorotUniform(),
#     )
#     l_pool1 = lasagne.layers.MaxPool1DLayer(
#         l_conv1,
#         ds=2,
#     )
#     l_conv2 = lasagne.layers.Conv1DLayer(
#         l_pool1,
#         num_filters=128,
#         filter_length=3,
#         stride=1,
#         border_mode='same',
#         nonlinearity=lasagne.nonlinearities.rectify,
#         W=lasagne.init.GlorotUniform(),
#     )
#     l_pool2 = lasagne.layers.MaxPool1DLayer(
#         l_conv2,
#         ds=2,
#     )
#     l_conv3 = lasagne.layers.Conv1DLayer(
#         l_pool2,
#         num_filters=100,
#         filter_length=3,
#         stride=1,
#         border_mode='same',
#         nonlinearity=lasagne.nonlinearities.rectify,
#         W=lasagne.init.GlorotUniform(),
#     )
#     l_pool3 = lasagne.layers.MaxPool1DLayer(
#         l_conv3,
#         ds=2,
#     )
#     l_conv4 = lasagne.layers.Conv1DLayer(
#         l_pool3,
#         num_filters=100,
#         filter_length=3,
#         stride=1,
#         border_mode='same',
#         nonlinearity=lasagne.nonlinearities.rectify,
#         W=lasagne.init.GlorotUniform(),
#     )
#     l_pool4 = lasagne.layers.MaxPool1DLayer(
#         l_conv4,
#         ds=2,
#     )
#     l_conv5 = lasagne.layers.Conv1DLayer(
#         l_pool4,
#         num_filters=200,
#         filter_length=3,
#         stride=1,
#         border_mode='same',
#         nonlinearity=lasagne.nonlinearities.rectify,
#         W=lasagne.init.GlorotUniform(),
#     )
#     l_pool5 = lasagne.layers.MaxPool1DLayer(
#         l_conv5,
#         ds=2,
#     )
#     l_conv6 = lasagne.layers.Conv1DLayer(
#         l_pool5,
#         num_filters=200,
#         filter_length=3,
#         stride=1,
#         border_mode='same',
#         nonlinearity=lasagne.nonlinearities.rectify,
#         W=lasagne.init.GlorotUniform(),
#     )
#     l_pool6 = lasagne.layers.MaxPool1DLayer(
#         l_conv6,
#         ds=2,
#     )
    l_fc1 = lasagne.layers.DenseLayer(
        l_in,
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
 
