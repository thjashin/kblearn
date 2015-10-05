#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import dnn


def unit_norm(x):
    """Unit-Normalization activation function

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 where the 2-norm of the row is 1 and each single value is in [0, 1]
        The output of the unit_norm function applied to the activation.
    """
    eps = 1e-6  # avoid NaN gradient when T.sqrt(0)
    return (x.T / T.sqrt((x ** 2).T.sum(axis=0) + eps)).T


def build_model(input_dim, output_dim, word_vectors, batch_size=None):
    V, D = word_vectors.shape
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 1, input_dim),
    )
    l_emb = lasagne.layers.EmbeddingLayer(
        l_in,
        input_size=V,
        output_size=D,
        W=word_vectors.astype('float32'),
    )
    l_transpose = lasagne.layers.DimshuffleLayer(
        l_emb,
        pattern=(0, 1, 3, 2),
    )
    l_conv11 = dnn.Conv2DDNNLayer(
        l_transpose,
        num_filters=64,
        filter_size=(D, 1),
        stride=(D, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_conv12 = dnn.Conv2DDNNLayer(
        l_conv11,
        num_filters=64,
        filter_size=(1, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool1 = dnn.MaxPool2DDNNLayer(
        l_conv12,
        pool_size=(1, 2),
    )
    l_conv21 = dnn.Conv2DDNNLayer(
        l_pool1,
        num_filters=128,
        filter_size=(1, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_conv22 = dnn.Conv2DDNNLayer(
        l_conv21,
        num_filters=128,
        filter_size=(1, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool2 = dnn.MaxPool2DDNNLayer(
        l_conv22,
        pool_size=(1, 2),
    )
    l_conv3 = dnn.Conv2DDNNLayer(
        l_pool2,
        num_filters=256,
        filter_size=(1, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool3 = dnn.MaxPool2DDNNLayer(
        l_conv3,
        pool_size=(1, 2),
    )
    l_conv4 = dnn.Conv2DDNNLayer(
        l_pool3,
        num_filters=512,
        filter_size=(1, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
    )
    l_pool4 = dnn.MaxPool2DDNNLayer(
        l_conv4,
        pool_size=(1, 2),
    )
    #     l_conv5 = dnn.Conv2DDNNLayer(
    #         l_pool4,
    #         num_filters=512,
    #         filter_size=(1, 3),
    #         stride=(1, 1),
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         W=lasagne.init.GlorotUniform(),
    #     )
    #     l_pool5 = dnn.MaxPool2DDNNLayer(
    #         l_conv5,
    #         pool_size=(1, 2),
    #     )
    l_fc1 = lasagne.layers.DenseLayer(
        l_pool4,
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
        nonlinearity=unit_norm,
        W=lasagne.init.Uniform(0.001),
    )
    return l_out


def SemanticFunc(sem_model):
    input_ = T.itensor3('input_')
    output = lasagne.layers.get_output(sem_model, inputs=input_, deterministic=True)
    return theano.function([input_], [output])
