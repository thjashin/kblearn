#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import lasagne


def build_model(input_dim, output_dim, batch_size):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=500,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    # TODO(jiaxin): add a normalization layer
    return l_out
