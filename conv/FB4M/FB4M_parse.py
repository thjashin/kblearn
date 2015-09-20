import os
import sys
import re
import cPickle
import gc
import scipy.sparse as sp
import word2vec

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
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

entity2idx = {}
idx2entity = {}

#############################################################################
### Creation of the word dictionaries for text

pjoin = os.path.join
pdir = os.path.dirname
pabs = os.path.abspath
ENTITY_DESCRIPTION_DATA = pjoin(datapath, 'topic-description-has-desc.tsv')

items = []
with open(ENTITY_DESCRIPTION_DATA, 'r') as f:
    text = f.read().decode('utf8').strip().split('\n')
for line in text:
    arr = line.split('\t')
    item = {'mid': arr[0].strip(), 'description': arr[1].strip()}
    items.append(item)

stopwords_ = set(stopwords.words())
# punctuations = set(string.punctuation)

word2id = {}
id2word = {}
items_seg = []
lens = []
for item in items:
    mid = item['mid']
    desc = item['description']

    if mid in entity2idx:
        continue
    idx = entity2idx.setdefault(mid, len(entity2idx))
    idx2entity.setdefault(idx, mid)

    # letters only
    desc = re.sub("[^a-zA-Z]", " ", desc)

    # discuss: whether to deal with stopwords
    words = word_tokenize(desc.lower())
    wids = []
    # words = filter(lambda x: x not in punctuations, words)
    words = filter(lambda x: x not in stopwords_, words)
    for word in set(words):
        id_ = word2id.setdefault(word, len(word2id))
        id2word.setdefault(id_, word)
        wids.append(id_)

    lens.append(len(words))
    items_seg.append(wids)

print 'len(items_seg):', len(items_seg)
print 'len(entity2idx):', len(entity2idx)

del items
del text
gc.collect()

print 'len(word2id):', len(word2id)
print 'len(id2word):', len(id2word)

with open('../data/FB4M_word2id.pkl', 'w') as f:
    cPickle.dump(word2id, f, -1)
with open('../data/FB4M_id2word.pkl', 'w') as f:
    cPickle.dump(id2word, f, -1)

with open('../data/FB4M_concat-words.txt', 'w') as f:
    f.write('\n'.join([' '.join(map(str, i)) for i in items_seg]))

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

concat_words_cut = np.ones((len(items_seg), max_len), dtype='int') * (len(word2id))
for i, wids in enumerate(items_seg):
    concat_words_cut[i, :min(len(wids), max_len)] = wids[:max_len]
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

#################################################
### Creation of the entities/indices dictionnaries

np.random.seed(753)

entleftlist = []
entrightlist = []
rellist = []

for datatyp in ['train', 'valid', 'test']:
    f = open(rand_split_path + '%s.tsv' % datatyp, 'r')
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

nbleft = len(entleftset)
nbshared = len(entsharedset)
nbright = len(entrightset)
print "# of only_left/shared/only_right entities: ", nbleft, '/', nbshared, '/', nbright
# add relations at the end of the dictionary
idx = len(entity2idx)
for i in relset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbrel = idx - len(entity2idx)
print "Number of relations: ", nbrel

f = open('../data/FB4M_entity2idx.pkl', 'w')
g = open('../data/FB4M_idx2entity.pkl', 'w')
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
    f = open('../data/FB4M-%s-lhs.pkl' % datatyp, 'w')
    g = open('../data/FB4M-%s-rhs.pkl' % datatyp, 'w')
    h = open('../data/FB4M-%s-rel.pkl' % datatyp, 'w')
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

# for i in remove_tst_ex:
#     print i
