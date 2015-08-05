#! /usr/bin/python
from FB4M_exp import *

launch(
    op='TransE',
    simfn='L1',
    ndim=50,
    nhid=50,
    margin=1.,
    lrweights=1e-6,
    momentum=0.9,
    lremb=0.01,
    lrparam=1. / 4000,
    nbatches=100,
    totepochs=1000,
    test_all=1,
    neval=1000,
    savepath='FB4M_TransE',
    datapath='../data/',
    eval_batchsize=40000,
    entity_batchsize=40000
)
