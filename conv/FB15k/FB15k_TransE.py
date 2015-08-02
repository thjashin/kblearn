#! /usr/bin/python
from FB15k_exp import *

launch(op='TransE', simfn='L1', ndim=50, nhid=50, marge=1., lrweights=1e-6, momentum=0.9, lremb=0.01, lrparam=1.0 / 4831,
    nbatches=4000, printbatches=40, totepochs=1000, test_all=10, neval=1000, savepath='FB15k_TransE', datapath='../data/')

