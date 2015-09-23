import os
import re
import gc
import sys
import cPickle
from collections import defaultdict
import scipy.sparse as sp

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def save_sparse_csr(filename, array):
    np.savez_compressed(filename, data=array.data, indices=array.indices,
                        indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


# Put the freebase4M data absolute path here
datapath = '/mfs/jiaxin/data/freebase-has-desc/'
rand_split_path = os.path.join(datapath, 'entity-split') + '/'

assert datapath is not None

if 'data' not in os.listdir('../'):
    os.mkdir('../data')


def parseline(line):
    lhs, rel, rhs = line.strip().split('\t')
    return lhs.strip(), rel.strip(), rhs.strip()

entity2idx = {}
idx2entity = {}

#############################################################################
### Creation of the word dictionaries and bag-of-words feature for text

print 'loading text...'
sys.stdout.flush()

pjoin = os.path.join
pdir = os.path.dirname
pabs = os.path.abspath
ENTITY_DESCRIPTION_DATA = pjoin(datapath, 'topic-description-has-desc.tsv')

stopwords_ = set(stopwords.words())
# punctuations = set(string.punctuation)
items_seg = {}

with open(ENTITY_DESCRIPTION_DATA, 'r') as f:
    text = f.read().decode('utf8').strip().split('\n')
for line in text:
    arr = line.strip().split('\t')
    mid = arr[0].strip()
    desc = arr[1].strip()
    # letters only
    desc = re.sub("[^a-zA-Z]", " ", desc)
    words = word_tokenize(desc.lower())
    # words = filter(lambda x: x not in punctuations, words)
    words = filter(lambda x: x not in stopwords_, words)

    items_seg[mid] = words

del text
gc.collect()

#################################################
### Creation of the entities/indices dictionnaries

print 'dealing with entities...'
sys.stdout.flush()

np.random.seed(753)

rellist = set()
entities = defaultdict(set)
entleftlist = set()
entrightlist = set()

for datatyp in ['train', 'valid', 'test']:
    f = open(rand_split_path + '%s.tsv' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    for i in dat:
        lhs, rel, rhs = parseline(i)
        entleftlist.add(lhs)
        entrightlist.add(rhs)
        rellist.add(rel)
        entities[datatyp].add(lhs)
        entities[datatyp].add(rhs)

print 'train/valid/test/total entities:', \
    len(entities['train']), len(entities['valid']), len(entities['test']), \
    len(entities['train'] | entities['valid'] | entities['test'])

entleftset = entleftlist - entrightlist
entsharedset = entleftlist & entrightlist
entrightset = entrightlist - entleftlist

nbleft = len(entleftset)
nbshared = len(entsharedset)
nbright = len(entrightset)
print "# of only_left/shared/only_right/total entities: ", \
    nbleft, '/', nbshared, '/', nbright, '/', nbleft + nbshared + nbright

for datatyp in ['train', 'valid', 'test']:
    for entity in entities[datatyp]:
        idx = entity2idx.setdefault(entity, len(entity2idx))
        idx2entity.setdefault(idx, entity)
print 'Number of entities:', len(entity2idx)

# add relations at the end of the dictionary
idx = len(entity2idx)
Nsyn = idx
for i in rellist:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbrel = idx - Nsyn
print "Number of relations:", nbrel

f = open('../data/FB4M_entity2idx.pkl', 'w')
g = open('../data/FB4M_idx2entity.pkl', 'w')
cPickle.dump(entity2idx, f, -1)
cPickle.dump(idx2entity, g, -1)
f.close()
g.close()

print 'len(entity2idx):', len(entity2idx)
print 'len(idx2entity):', len(idx2entity)

#################################################
### Creation of the dataset files

print 'dump triples...'
sys.stdout.flush()

for datatyp in ['train', 'valid', 'test']:
    print datatyp

    f = open(rand_split_path + '%s.tsv' % datatyp, 'r')
    dat = f.readlines()
    f.close()

    # Declare the dataset variables
    inpl = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
                         dtype='float32')
    inpr = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
                         dtype='float32')
    inpo = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)),
                         dtype='float32')

    # Fill the sparse matrices
    ct = 0
    for i in dat:
        lhs, rel, rhs = parseline(i)
        inpl[entity2idx[lhs], ct] = 1
        inpr[entity2idx[rhs], ct] = 1
        inpo[entity2idx[rel], ct] = 1
        ct += 1

    # Save the datasets
    if 'data' not in os.listdir('../'):
        os.mkdir('../data')
    f = open('../data/FB4M-%s-lhs.pkl' % datatyp, 'w')
    g = open('../data/FB4M-%s-rhs.pkl' % datatyp, 'w')
    h = open('../data/FB4M-%s-rel.pkl' % datatyp, 'w')
    inpl_csr = inpl.tocsr()
    cPickle.dump(inpl_csr, f, -1)
    cPickle.dump(inpr.tocsr(), g, -1)
    cPickle.dump(inpo.tocsr(), h, -1)
    f.close()
    g.close()
    h.close()

    print 'inpl.shape:', inpl_csr.shape
    print 'inpl.nonzeros:', inpl_csr.nonzero()[0].shape

############################################################################
### Creation of the n-gram dictionaries and bag-of-ngrams feature for text
### ngrams are generated by word with start and ending marks (#)

print 'dump text features...'
sys.stdout.flush()

n = 3
indptr = [0]
indices = []
data = []

ngram2id = {}
id2ngram = {}
for idx in xrange(len(entity2idx) - nbrel):
    mid = idx2entity[idx]
    words = items_seg[mid]
    for word in set(words):
        word = '#%s#' % word
        for i in xrange(len(word) - n + 1):
            ngram = word[i:(i + n)]
            id_ = ngram2id.setdefault(ngram, len(ngram2id))
            id2ngram.setdefault(id_, ngram)
            indices.append(id_)
            data.append(1)
    indptr.append(len(indices))

entity_ngrams = sp.csr_matrix((data, indices, indptr), dtype='float32').log1p()
print 'entity_ngrams.shape:', entity_ngrams.shape

# for k in sorted(ngram2id.keys()):
#     print k,
# print

print 'len(ngram2id):', len(ngram2id)
print 'len(id2ngram):', len(id2ngram)

with open('../data/FB4M_%dgram2id.pkl' % n, 'w') as f:
    cPickle.dump(ngram2id, f, -1)
with open('../data/FB4M_id2%dgram.pkl' % n, 'w') as f:
    cPickle.dump(id2ngram, f, -1)
save_sparse_csr('../data/FB4M-bag-of-%dgrams.npz' % n, entity_ngrams)
