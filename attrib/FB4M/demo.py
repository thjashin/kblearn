#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import re
import sys
import string

import theano
import numpy as np
import scipy
from scipy import sparse as sp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from model import BatchRankLeftFnIdx, BatchRankRightFnIdx, RankFnIdx
from semantic import SemanticFunc


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
                                   dtype=theano.config.floatX)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def init(loadmodel, datapath='../data/', dataset='FB4M', Nsyn=4661857,
         entity_batchsize=80000):
    f = open(loadmodel + '.ents')
    sem_model = cPickle.load(f)
    embeddings = cPickle.load(f)
    entity_embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()

    # load ngram features of entities
    with open(datapath + dataset + '_id23gram.pkl', 'r') as f:
        id2ngram = cPickle.load(f)
    with open(datapath + dataset + '_3gram2id.pkl', 'r') as f:
        ngram2id = cPickle.load(f)

    with open(datapath + dataset + '_idx2entity.pkl', 'r') as f:
        idx2entity = cPickle.load(f)

    entity_ngrams = load_sparse_csr(datapath + dataset +
                                    '-bag-of-3grams.npz').astype(theano.config.floatX)
    print 'entity_ngrams.shape:', entity_ngrams.shape

    sem_func = SemanticFunc(sem_model)
    batch_ranklfunc = BatchRankLeftFnIdx(simfn, embeddings, leftop, rightop,
                                         subtensorspec=Nsyn)
    batch_rankrfunc = BatchRankRightFnIdx(simfn, embeddings, leftop, rightop,
                                          subtensorspec=Nsyn)

    # get sem output for all entities
    # n_entity_batches = Nsyn / entity_batchsize
    #
    # entity_embeddings = []
    # for i in xrange(n_entity_batches):
    #     entity_embeddings.append(
    #         sem_func(entity_ngrams[i * entity_batchsize:(i + 1) * entity_batchsize].toarray())[0]
    #     )
    # if n_entity_batches * entity_batchsize < Nsyn:
    #     entity_embeddings.append(
    #         sem_func(entity_ngrams[n_entity_batches * entity_batchsize:].toarray())[0]
    #     )
    # entity_embeddings = np.vstack(entity_embeddings)

    # with open(loadmodel + '.ents', 'w') as f:
    #     cPickle.dump(sem_model, f, -1)
    #     cPickle.dump(embeddings, f, -1)
    #     cPickle.dump(entity_embeddings, f, -1)
    #     cPickle.dump(leftop, f, -1)
    #     cPickle.dump(rightop, f, -1)
    #     cPickle.dump(simfn, f, -1)

    return ngram2id, id2ngram, batch_rankrfunc, sem_func, entity_embeddings, idx2entity


def query(desc, ngram2id, id2ngram, batch_func, sem_func, embeddings,
          idx2entity, Nsyn=4661857, Nrel=2663, eval_batchsize=5120000):
    # translate into n-grams
    stopwords_ = set(stopwords.words())
    punctuations = set(string.punctuation)

    # letters only
    desc = re.sub("[^a-zA-Z]", " ", desc)

    # discuss: whether to deal with stopwords
    words = word_tokenize(desc.lower())
    # words = filter(lambda x: x not in punctuations, words)
    words = filter(lambda x: x not in stopwords_, words)

    # 3gram
    n = 3
    ngram_onehot = np.zeros(len(ngram2id), dtype='float32')
    for word in set(words):
        word = '#%s#' % word
        for i in xrange(len(word) - n + 1):
            ngram = word[i:(i + n)]
            if ngram in ngram2id:
                ngram_onehot[ngram2id[ngram]] += 1.0
    ngram_onehot = np.log1p(ngram_onehot).reshape(1, len(ngram2id))

    lhs = sem_func(ngram_onehot)[0].squeeze()
    total = []
    for idxo in xrange(Nrel):
        sys.stdout.write('%d\r' % idxo)
        sys.stdout.flush()
        sims = RankFnIdx(batch_func, embeddings, lhs, idxo, eval_batchsize)[0]
        largest_sims_ind = sims.argpartition(-10)[-10:]
        total += zip(sims[largest_sims_ind], [idxo * Nsyn + i for i in largest_sims_ind])

    for sim, i in reversed(sorted(total)[-10:]):
        print sim, idx2entity[Nsyn + i / Nsyn], idx2entity[i % Nsyn]


if __name__ == "__main__":
    context = init(sys.argv[1])
    while True:
        input_ = raw_input("Describe a concept:")
        if input_.lower() == "exit":
            break
        query(input_, *context)
