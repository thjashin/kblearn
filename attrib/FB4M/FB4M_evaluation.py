#! /usr/bin/python
import sys
import cPickle

import scipy
from scipy import sparse as sp
import numpy as np
import theano

from model import *
from semantic import SemanticFunc


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
                                   dtype=theano.config.floatX)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


def RankingEval(datapath='../data/', dataset='FB4M',
                loadmodel='best_valid_model.pkl', neval=80000, Nsyn=4562841, n=10,
                entity_batchsize=80000, eval_batchsize=5120000):
    # Load model
    f = open(loadmodel)
    sem_model = cPickle.load(f)
    embeddings = cPickle.load(f)
    # entity_embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()

    with open(datapath + dataset + '_idx2entity.pkl', 'r') as f:
        idx2entity = cPickle.load(f)

    # load ngram features of entities
    entity_ngrams = load_sparse_csr(datapath + dataset +
                                    '-bag-of-3grams.npz').astype(theano.config.floatX)
    print 'entity_ngrams.shape:', entity_ngrams.shape

    # Load data
    l = load_file(datapath + dataset + '-test-lhs.pkl')
    r = load_file(datapath + dataset + '-test-rhs.pkl')
    o = load_file(datapath + dataset + '-test-rel.pkl')
    if type(embeddings) is list:
        o = o[-embeddings[1].N:, :]

    # Convert sparse matrix to indexes
    if neval == 'all':
        idxl = convert2idx(l)
        idxr = convert2idx(r)
        idxo = convert2idx(o)
    else:
        idxl = convert2idx(l)[:neval]
        idxr = convert2idx(r)[:neval]
        idxo = convert2idx(o)[:neval]

    sem_func = SemanticFunc(sem_model)
    batch_ranklfunc = BatchRankLeftFnIdx(simfn, embeddings, leftop, rightop,
                                         subtensorspec=Nsyn)
    batch_rankrfunc = BatchRankRightFnIdx(simfn, embeddings, leftop, rightop,
                                          subtensorspec=Nsyn)

    # get sem output for all entities
    n_entity_batches = Nsyn / entity_batchsize

    entity_embeddings = []
    for i in xrange(n_entity_batches):
        entity_embeddings.append(
            sem_func(entity_ngrams[i * entity_batchsize:(i + 1) * entity_batchsize].toarray())[0]
        )
    if n_entity_batches * entity_batchsize < Nsyn:
        entity_embeddings.append(
            sem_func(entity_ngrams[n_entity_batches * entity_batchsize:].toarray())[0]
        )
    entity_embeddings = np.vstack(entity_embeddings)

    with open(loadmodel + '.ents', 'w') as f:
        cPickle.dump(sem_model, f, -1)
        cPickle.dump(embeddings, f, -1)
        cPickle.dump(entity_embeddings, f, -1)
        cPickle.dump(leftop, f, -1)
        cPickle.dump(rightop, f, -1)
        cPickle.dump(simfn, f, -1)

    res = FastRankingScoreIdx(batch_ranklfunc, batch_rankrfunc,
                              entity_embeddings, idxl, idxr,
                              idxo, eval_batchsize)

    dres = {}
    dres.update({'microlmean': np.mean(res[0])})
    dres.update({'microlmedian': np.median(res[0])})
    dres.update({'microlhits@n': np.mean(np.asarray(res[0]) <= n) * 100})
    dres.update({'micrormean': np.mean(res[1])})
    dres.update({'micrormedian': np.median(res[1])})
    dres.update({'microrhits@n': np.mean(np.asarray(res[1]) <= n) * 100})
    resg = res[0] + res[1]
    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microghits@n': np.mean(np.asarray(resg) <= n) * 100})

    print "### MICRO:"
    print "\t-- left   >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
        n, round(dres['microlhits@n'], 3))
    print "\t-- right  >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
        n, round(dres['microrhits@n'], 3))
    print "\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
        n, round(dres['microghits@n'], 3))

    # write good candidates to file
    with open('hitat10.tsv', 'w') as f:
        for i in xrange(len(res[1])):
            if res[1][i] < 10:
                f.write('%d\t%s\t%s\t%s\n' % (
                    res[1][i], idx2entity[idxl[i]],
                    idx2entity[Nsyn + idxo[i]], idx2entity[idxr[i]]))


    listrel = set(idxo)
    dictrelres = {}
    dictrellmean = {}
    dictrelrmean = {}
    dictrelgmean = {}
    dictrellmedian = {}
    dictrelrmedian = {}
    dictrelgmedian = {}
    dictrellrn = {}
    dictrelrrn = {}
    dictrelgrn = {}

    for i in listrel:
        dictrelres.update({i: [[], []]})

    for i, j in enumerate(res[0]):
        dictrelres[idxo[i]][0] += [j]

    for i, j in enumerate(res[1]):
        dictrelres[idxo[i]][1] += [j]

    for i in listrel:
        dictrellmean[i] = np.mean(dictrelres[i][0])
        dictrelrmean[i] = np.mean(dictrelres[i][1])
        dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
        dictrellmedian[i] = np.median(dictrelres[i][0])
        dictrelrmedian[i] = np.median(dictrelres[i][1])
        dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
        dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
        dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
        dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] +
                                           dictrelres[i][1]) <= n) * 100

    dres.update({'dictrelres': dictrelres})
    dres.update({'dictrellmean': dictrellmean})
    dres.update({'dictrelrmean': dictrelrmean})
    dres.update({'dictrelgmean': dictrelgmean})
    dres.update({'dictrellmedian': dictrellmedian})
    dres.update({'dictrelrmedian': dictrelrmedian})
    dres.update({'dictrelgmedian': dictrelgmedian})
    dres.update({'dictrellrn': dictrellrn})
    dres.update({'dictrelrrn': dictrelrrn})
    dres.update({'dictrelgrn': dictrelgrn})

    dres.update({'macrolmean': np.mean(dictrellmean.values())})
    dres.update({'macrolmedian': np.mean(dictrellmedian.values())})
    dres.update({'macrolhits@n': np.mean(dictrellrn.values())})
    dres.update({'macrormean': np.mean(dictrelrmean.values())})
    dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
    dres.update({'macrorhits@n': np.mean(dictrelrrn.values())})
    dres.update({'macrogmean': np.mean(dictrelgmean.values())})
    dres.update({'macrogmedian': np.mean(dictrelgmedian.values())})
    dres.update({'macroghits@n': np.mean(dictrelgrn.values())})

    print "### MACRO:"
    print "\t-- left   >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
        n, round(dres['macrolhits@n'], 3))
    print "\t-- right  >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
        n, round(dres['macrorhits@n'], 3))
    print "\t-- global >> mean: %s, median: %s, hits@%s: %s%%" % (
        round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
        n, round(dres['macroghits@n'], 3))

    return dres


if __name__ == '__main__':
    RankingEval(loadmodel=sys.argv[1])
