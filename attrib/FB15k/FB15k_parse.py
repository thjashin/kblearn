import os, sys
import cPickle

import numpy as np
import scipy.sparse as sp

# Put the freebase15k data absolute path here
datapath = '/home/cc/jxshi/data/FB15k/'
assert datapath is not None

if 'data' not in os.listdir('../'):
    os.mkdir('../data')


def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

#################################################
### Creation of the entities/indices dictionnaries

np.random.seed(753)

entleftlist = []
entrightlist = []
rellist = []

for datatyp in ['train']:
    f = open(datapath + 'freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        entleftlist += [lhs[0]]
        entrightlist += [rhs[0]]
        rellist += [rel[0]]

entleftset = np.sort(list(set(entleftlist) - set(entrightlist)))
entsharedset = np.sort(list(set(entleftlist) & set(entrightlist)))
entrightset = np.sort(list(set(entrightlist) - set(entleftlist)))
relset = np.sort(list(set(rellist)))

entity2idx = {}
idx2entity = {}


# we keep the entities specific to one side of the triplets contiguous
idx = 0
for i in entrightset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbright = idx
for i in entsharedset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbshared = idx - nbright
for i in entleftset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbleft = idx - (nbshared + nbright)

print "# of only_left/shared/only_right entities: ", nbleft, '/', nbshared, '/', nbright
# add relations at the end of the dictionary
for i in relset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbrel = idx - (nbright + nbshared + nbleft)
print "Number of relations: ", nbrel

f = open('../data/FB15k_entity2idx.pkl', 'w')
g = open('../data/FB15k_idx2entity.pkl', 'w')
cPickle.dump(entity2idx, f, -1)
cPickle.dump(idx2entity, g, -1)
f.close()
g.close()

#################################################
### Creation of the dataset files

unseen_ents=[]
remove_tst_ex=[]

for datatyp in ['train', 'valid', 'test']:
    print datatyp
    f = open(datapath + 'freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
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
        lhs, rel, rhs = parseline(i[:-1])
        if lhs[0] in entity2idx and rhs[0] in entity2idx and rel[0] in entity2idx: 
            inpl[entity2idx[lhs[0]], ct] = 1
            inpr[entity2idx[rhs[0]], ct] = 1
            inpo[entity2idx[rel[0]], ct] = 1
            ct += 1
        else:
            if lhs[0] in entity2idx:
                unseen_ents+=[lhs[0]]
            if rel[0] in entity2idx:
                unseen_ents+=[rel[0]]
            if rhs[0] in entity2idx:
                unseen_ents+=[rhs[0]]
            remove_tst_ex+=[i[:-1]]

    # Save the datasets
    if 'data' not in os.listdir('../'):
        os.mkdir('../data')
    f = open('../data/FB15k-%s-lhs.pkl' % datatyp, 'w')
    g = open('../data/FB15k-%s-rhs.pkl' % datatyp, 'w')
    h = open('../data/FB15k-%s-rel.pkl' % datatyp, 'w')
    cPickle.dump(inpl.tocsr(), f, -1)
    cPickle.dump(inpr.tocsr(), g, -1)
    cPickle.dump(inpo.tocsr(), h, -1)
    f.close()
    g.close()
    h.close()

unseen_ents=list(set(unseen_ents))
print len(unseen_ents)
remove_tst_ex=list(set(remove_tst_ex))
print len(remove_tst_ex)

for i in remove_tst_ex:
    print i

#############################################################################
### Creation of the word dictionaries and bag-of-words feature for text

pjoin = os.path.join
pdir = os.path.dirname
ENTITY_DESCRIPTION_DATA = pjoin(pdir(pdir(__file__)), 'crawfb', 'items.json')

with open(ENTITY_DESCRIPTION_DATA, 'r') as f:
    items = json.load(f)


word2id = {}
id2word = {}
items_seg = []
for item in items:
    mid = item['mid']
    name = item['name']
    desc = item['description']

    # discuss: whether to deal with stopwords
    words = word_tokenize(desc)
    for word in set(words):
        if word not in word2id:
            id_ = len(word2id)
            word2id[word] = id_
            id2word[id_] = word

    items_seg.append((mid, name, words))

with open('../data/FB15k_word2id.pkl', 'w') as f:
    cPickle.dump(word2id, f, -1)
with open('../data/FB15k_id2word.pkl', 'w') as f:
    cPickle.dump(id2word, f, -1)


entity_words = sp.lil_matrix(
    (np.max(entity2idx.values()) + 1, len(word2id)), dtype='float32')
for mid, name, words in items_segs:
    if mid in entity2idx:
        id_ = entity2idx[mid]
        for word in words:
            entity_words[id_, word2id[word]] += 1

with open('../data/FB15k-bag-of-words.pkl', 'w') as f:
    cPickle.dump(entity_words.to_csr(), f, -1)

#############################################################################
### Creation of the n-gram dictionaries and bag-of-ngrams feature for text
### ngrams are generated by word with start and ending marks (#)

n = 3
ngram2id = {}
id2ngram = {}
for mid, name, words in item_segs:
    for word in set(words):
        word = '#%s#' % word
        ngrams = [word[i:(i+3)] for i in xrange(len(word) - 2)]
        for ngram in ngrams:
            id_ = len(ngram2id)
            ngram2id[ngram] = id_
            id2ngram[id_] = ngram

with open('../data/FB15k_%dgram2id.pkl' % n, 'w') as f:
    cPickle.dump(ngram2id, f, -1)
with open('../data/FB15k_id2%dgram.pkl' % n, 'w') as f:
    cPickle.dump(id2ngram, f, -1)


entity_ngrams_dic = defaultdict(int)
entity_ngrams = sp.lil_matrix(
    (np.max(entity2idx.values()) + 1, len(ngram2id)), dtype='float32')

for mid, name, words in item_segs:
    id_ = entity2idx[mid]
    for word in words:
        word = '#%s#' % word
        ngrams = [word[i:(i+3)] for i in xrange(len(word) - 2)]
        for ngram in ngrams:
            entity_ngrams_dic[(id_, ngram2id[ngram])] += 1

for k, v in entity_ngrams_dic.iteritems():
    entity_ngrams[k[0], k[1]] = v

with open('../data/FB15k-bag-of-ngrams.pkl', 'w') as f:
    cPickle.dump(entity_ngrams.to_csr(), f, -1)





