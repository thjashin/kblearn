#! /usr/bin/python
from FB15k_exp import *

launch(op='TransE', simfn='L1', ndim=50, nhid=50, marge=1., lrweights=3e-7, momentum=0.9, lremb=0.01, lrparam=1.,
    nbatches=100, totepochs=1000, test_all=10, neval=1000, savepath='FB15k_TransE', datapath='../data/')

