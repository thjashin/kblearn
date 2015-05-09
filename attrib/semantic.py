#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import lasagne


def build_model(input_dim, output_dim, batch_size=None):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=500,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    #l_hidden1_dropout = lasagne.layers.DropoutLayer(
    #    l_hidden1,
    #    p=0.5,
    #)
    #l_hidden2 = lasagne.layers.DenseLayer(
    #    l_hidden1,
    #    num_units=500,
    #    nonlinearity=lasagne.nonlinearities.rectify,
    #    W=lasagne.init.GlorotUniform(),
    #)
    #l_hidden2_dropout = lasagne.layers.DropoutLayer(
    #    l_hidden2,
    #    p=0.5,
    #)
    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    # TODO(jiaxin): add a normalization layer
    return l_out


def build_conv_model(input_dim, output_dim, batch_size=None):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    l_conv1 = lasagne.layers.Conv1DLayer(
        l_in,
        num_filters=20,
        filter_size=(3,),
        stride=(1,),
        border_mode='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool1 = lasagne.layers.Pool1DLayer(
        l_conv1,
        pool_size=(2,),
    )
    l_conv2 = lasagne.layers.Conv1DLayer(
        l_pool1,
        num_filters=40,
        stride=(1,),
        border_mode='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool2 = lasagne.layers.Pool1DLayer(
        l_conv2,
        pool_size=(2,),
    )
    l_conv3 = lasagne.layers.Conv1DLayer(
        l_pool3,
        num_filters=100,
        stride=(1,),
        border_mode='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool3 = lasagne.layers.Pool1DLayer(
        l_conv3,
        pool_size=(2,),
    )
    l_conv4 = lasagne.layers.Conv1DLayer(
        l_pool3,
        num_filters=100,
        stride=(1,),
        border_mode='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool4 = lasagne.layers.Pool1DLayer(
        l_conv4,
        pool_size=(2,),
    )
    l_conv5 = lasagne.layers.Conv1DLayer(
        l_pool4,
        num_filters=200,
        stride=(1,),
        border_mode='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool5 = lasagne.layers.Pool1DLayer(
        l_conv5,
        pool_size=(2,),
    )
    l_conv6 = lasagne.layers.Conv1DLayer(
        l_pool5,
        num_filters=200,
        stride=(1,),
        border_mode='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool6 = lasagne.layers.Pool1DLayer(
        l_conv6,
        pool_size=(2,),
    )
    l_fc1 = lasagne.layers.DenseLayer(
        l_pool6,
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
 
