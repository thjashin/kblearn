import os
import re
import json
import string
import cPickle
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import word2vec



# Put the freebase15k data absolute path here
datapath = '/home/jiaxin/mfs/data/FB15k/'
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

unseen_ents = []
remove_tst_ex = []

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
                unseen_ents += [lhs[0]]
            if rel[0] in entity2idx:
                unseen_ents += [rel[0]]
            if rhs[0] in entity2idx:
                unseen_ents += [rhs[0]]
            remove_tst_ex += [i[:-1]]

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

unseen_ents = list(set(unseen_ents))
print len(unseen_ents)
remove_tst_ex = list(set(remove_tst_ex))
print len(remove_tst_ex)

for i in remove_tst_ex:
    print i

#############################################################################
### Creation of the word dictionaries and bag-of-words feature for text

pjoin = os.path.join
pdir = os.path.dirname
pabs = os.path.abspath
ENTITY_DESCRIPTION_DATA = pjoin(pdir(pdir(pdir(pabs(__file__)))),
                                'crawlfb', 'items.json')

with open(ENTITY_DESCRIPTION_DATA, 'r') as f:
    items = json.load(f)

stopwords_ = set(stopwords.words())
punctuations = set(string.punctuation)
word2id = {}
id2word = {}
items_seg = []
for item in items:
    mid = item['mid']
    name = item['name']
    desc = item['description']

    # letters only
    desc = re.sub("[^a-zA-Z]", " ", desc)

    # discuss: whether to deal with stopwords
    words = word_tokenize(desc.lower())
    # words = filter(lambda x: x not in punctuations, words)
    words = filter(lambda x: x not in stopwords_, words)

    for word in set(words):
        if word not in word2id:
            id_ = len(word2id)
            word2id[word] = id_
            id2word[id_] = word

    items_seg.append((mid, name, words))

print 'len(word2id):', len(word2id)
print 'len(id2word):', len(id2word)

with open('../data/FB15k_word2id.pkl', 'w') as f:
    cPickle.dump(word2id, f, -1)
with open('../data/FB15k_id2word.pkl', 'w') as f:
    cPickle.dump(id2word, f, -1)

entity_words = sp.lil_matrix(
    (np.max(entity2idx.values()) + 1, len(word2id)), dtype='float32')
for mid, name, words in items_seg:
    if mid in entity2idx:
        id_ = entity2idx[mid]
        for word in words:
            entity_words[id_, word2id[word]] += 1

print 'entity_words.shape:', entity_words.shape

with open('../data/FB15k-bag-of-words.pkl', 'w') as f:
    cPickle.dump(entity_words.tocsr(), f, -1)

#############################################################################
### Creation of the n-gram dictionaries and bag-of-ngrams feature for text
### ngrams are generated by word with start and ending marks (#)

n = 3
ngram2id = {}
id2ngram = {}
for mid, name, words in items_seg:
    for word in set(words):
        word = '#%s#' % word
        ngrams = [word[i:(i + n)] for i in xrange(len(word) - n + 1)]
        for ngram in ngrams:
            if ngram not in ngram2id:
                id_ = len(ngram2id)
                ngram2id[ngram] = id_
                id2ngram[id_] = ngram

# for k in sorted(ngram2id.keys()):
#     print k,
# print

print 'len(ngram2id):', len(ngram2id)
print 'len(id2ngram):', len(id2ngram)

with open('../data/FB15k_%dgram2id.pkl' % n, 'w') as f:
    cPickle.dump(ngram2id, f, -1)
with open('../data/FB15k_id2%dgram.pkl' % n, 'w') as f:
    cPickle.dump(id2ngram, f, -1)

entity_ngrams_dic = defaultdict(int)
entity_ngrams = sp.lil_matrix(
    (np.max(entity2idx.values()) + 1, len(ngram2id)), dtype='float32')

ngramcnt = defaultdict(int)
for mid, name, words in items_seg:
    id_ = entity2idx[mid]
    for word in words:
        word = '#%s#' % word
        ngrams = [word[i:(i + n)] for i in xrange(len(word) - n + 1)]
        for ngram in ngrams:
            entity_ngrams_dic[(id_, ngram2id[ngram])] += 1
            ngramcnt[ngram] += 1

# for k, v in sorted(ngramcnt.items()):
#     print k, v

for k, v in entity_ngrams_dic.iteritems():
    entity_ngrams[k[0], k[1]] = np.log(v + 1.0)

print 'entity_ngrams.shape:', entity_ngrams.shape

with open('../data/FB15k-bag-of-%dgrams.pkl' % n, 'w') as f:
    cPickle.dump(entity_ngrams.tocsr(), f, -1)


###########################################################
### Creation of concatenate word vector feature for text

WORD_VEC_FILE = '/home/jiaxin/mfs/data/word2vec/vectors-50.bin'
wordvec = word2vec.load(WORD_VEC_FILE)
vectors = wordvec.vectors
vocab = wordvec.vocab
word2idx = {}
for i, w in enumerate(vocab):
    word2idx[w] = i

word_vec_dims = vectors[word2idx['king']].shape[0]
print 'word_vec_dims:', word_vec_dims

# Check coverage of word vectors on vocabulary
total = len(word2id)
cnt = 0
miss = []
for word in word2id:
    if word in word2idx:
        cnt += 1
    else:
        miss.append(word)
print 'word vector coverage ratio: %s, miss: %d/%d' % (cnt * 1.0 / total,
                                                       total - cnt, total)

# Check max/min length of input text
items_seg = []
lens = []
for item in items:
    mid = item['mid']
    name = item['name']
    desc = item['description']
    # letters only
    desc = re.sub("[^a-zA-Z]", " ", desc).lower()
    words = word_tokenize(desc.lower())
    lens.append(len(words))
    items_seg.append((mid, name, words))
print 'input text length: min(%d) / max(%d) / avg(%d) / median(%d)' % (
    min(lens), max(lens), np.mean(lens), np.median(lens))

limit_len = 240
max_len = min(max(lens), limit_len)
print 'feature length:', max_len
print 'covered samples:', np.sum(np.array(lens) <= max_len) * 1.0 / len(lens)

# with open('miss.txt', 'w') as f:
#     for word in miss:
#         f.write(word.encode('utf8') + '\n')

entity_inputs = np.zeros((np.max(entity2idx.values()) + 1, max_len * word_vec_dims),
                         dtype='float32')
for mid, name, words in items_seg:
    id_ = entity2idx[mid]
    if words:
        sen_vec = np.hstack([vectors[word2idx[w]] if w in word2idx
                             else np.zeros(300) for w in words]).astype('float32')[:max_len]
        entity_inputs[id_, :sen_vec.shape[0]] = sen_vec

np.savez_compressed('../data/FB15k-concat-word-vectors.npz', entity_inputs=entity_inputs)
