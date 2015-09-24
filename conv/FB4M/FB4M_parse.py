import os
import sys
import re
import cPickle
import gc
import scipy.sparse as sp
import word2vec
from collections import defaultdict

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Put the FB4M data absolute path here
datapath = '/home/jiaxin/mfs/data/freebase-has-desc/'
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
### Creation of the word dictionaries for text

print 'loading text...'
sys.stdout.flush()

pjoin = os.path.join
pdir = os.path.dirname
pabs = os.path.abspath
ENTITY_DESCRIPTION_DATA = pjoin(datapath, 'topic-description-has-desc.tsv')

stopwords_ = set(stopwords.words())
# punctuations = set(string.punctuation)
word2id = {}
id2word = {}
items_seg = {}
lens = []

with open(ENTITY_DESCRIPTION_DATA, 'r') as f:
    text = f.read().decode('utf8').strip().split('\n')
for line in text:
    arr = line.strip().split('\t')
    mid = arr[0].strip()
    if mid in items_seg:
        continue
    desc = ''
    if len(arr) == 2:
        desc = arr[1].strip()
    else:
        print 'bad line:', line.encode('utf8')
    # letters only
    desc = re.sub("[^a-zA-Z]", " ", desc)
    words = word_tokenize(desc.lower())
    wids = []
    # words = filter(lambda x: x not in punctuations, words)
    words = filter(lambda x: x not in stopwords_, words)
    for word in set(words):
        id_ = word2id.setdefault(word, len(word2id))
        id2word.setdefault(id_, word)
        wids.append(id_)

    lens.append(len(words))
    items_seg[mid] = wids

del text
gc.collect()

print 'len(items_seg):', len(items_seg)
print 'len(word2id):', len(word2id)
print 'len(id2word):', len(id2word)

with open('../data/FB4M_word2id.pkl', 'w') as f:
    cPickle.dump(word2id, f, -1)
with open('../data/FB4M_id2word.pkl', 'w') as f:
    cPickle.dump(id2word, f, -1)

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

###########################################################
### Creation of concatenate word feature for text

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
print 'input text length: min(%d) / max(%d) / avg(%d) / median(%d)' % (
    min(lens), max(lens), np.mean(lens), np.median(lens))

limit_len = 240
max_len = min(max(lens), limit_len)
print 'feature length:', max_len
print 'covered samples:', np.sum(np.array(lens) <= max_len) * 1.0 / len(lens)

concat_words_cut = np.ones((Nsyn, max_len), dtype='int') * (len(word2id))
for idx in xrange(Nsyn):
    entity = idx2entity[idx]
    wids = items_seg[entity]
    concat_words_cut[idx, :min(len(wids), max_len)] = wids[:max_len]
np.savez_compressed('../data/FB4M_concat-words.npz', entity_words=concat_words_cut)

print 'write FB4M_concat-words finished.'
sys.stdout.flush()

del items_seg
gc.collect()

# word embeddings
word_embeddings = np.random.normal(0, 0.01, (len(word2id) + 1, 50))
# last line for zero paddings
word_embeddings[-1, :] = np.zeros(50)
print 'word embeddings shape:', word_embeddings.shape
for word, id in word2id.iteritems():
    if word in word2idx:
        word_embeddings[id] = vectors[word2idx[word]]
np.savez_compressed('../data/FB4M-word-embeddings.npz',
                    word_embeddings=word_embeddings.astype('float32'))
print 'write FB4M word embeddings finished.'

# with open('miss.txt', 'w') as f:
#     for word in miss:
#         f.write(word.encode('utf8') + '\n')

# entity_inputs = np.zeros((np.max(entity2idx.values()) + 1, max_len * word_vec_dims),
#                          dtype='float32')
# for mid, words in items_seg:
#     id_ = entity2idx[mid]
#     if words:
#         sen_vec = np.hstack([vectors[word2idx[w]] if w in word2idx
#                              else np.zeros(50) for w in words[:max_len]]).astype('float32')
#         entity_inputs[id_, :sen_vec.shape[0]] = sen_vec
#
# np.savez_compressed('../data/FB4M-concat-word-vectors.npz', entity_inputs=entity_inputs)
# del entity_inputs
# gc.collect()

# print 'finished writing concatenate word vectors feature.'
