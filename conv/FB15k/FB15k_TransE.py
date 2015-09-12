#! /usr/bin/python
from FB15k_exp import *

launch(
    op='TransE',
    simfn='L1',
    ndim=50,
    nhid=50,
    margin=1.,
    lrweights=0.01,
    momentum=0.9,
    lremb=0.01,
    lrparam=0.01,
    nbatches=1000,
    printbatches=10,
    totepochs=1000,
    test_all=1,
    neval=1000,
    savepath='FB15k_TransE',
    datapath='../data/',
    eval_batchsize=20000,
    entity_batchsize=20000,
)
