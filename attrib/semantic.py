#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

import lasagne


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
    # l_hidden2 = lasagne.layers.DenseLayer(
    #    l_hidden1,
    #    num_units=500,
    #    nonlinearity=lasagne.nonlinearities.rectify,
    #    W=lasagne.init.GlorotUniform(),
    # )
    # l_hidden2_dropout = lasagne.layers.DropoutLayer(
    #    l_hidden2,
    #    p=0.5,
    # )
    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=unit_norm,
        W=lasagne.init.Uniform(0.001),
    )
    return l_out


def SemanticFunc(sem_model):
    input_ = T.matrix('input_')
    output = lasagne.layers.get_output(sem_model, inputs=input_, deterministic=True)
    return theano.function([input_], [output])
