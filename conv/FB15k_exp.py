#! /usr/bin/python
import os
import scipy
from scipy import sparse
import cPickle

import sys
import time
from model import *
from semantic import build_model, SemanticFunc


# Utils ----------------------------------------------------------------------
def create_random_mat(shape, listidx=None):
    """
    This function create a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    if listidx is None:
        listidx = np.arange(shape[0])
    listidx = listidx[np.random.permutation(len(listidx))]
    randommat = scipy.sparse.lil_matrix((shape[0], shape[1]),
                                        dtype=theano.config.floatX)
    idx_term = 0
    for idx_ex in range(shape[1]):
        if idx_term == len(listidx):
            idx_term = 0
        randommat[listidx[idx_term], idx_ex] = 1
        idx_term += 1
    return randommat.tocsr()


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
                                   dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z


def toidx(input_, n_eval):
    idxs = convert2idx(input_)
    return idxs, idxs[:n_eval]


# Experiment function --------------------------------------------------------
def FB15kexp(state, channel):
    # Show experiment parameters
    print >> sys.stderr, state
    np.random.seed(state.seed)

    # Experiment folder
    if hasattr(channel, 'remote_path'):
        state.savepath = channel.remote_path + '/'
    elif hasattr(channel, 'path'):
        state.savepath = channel.path + '/'
    else:
        if not os.path.isdir(state.savepath):
            os.mkdir(state.savepath)

    # load id2word
    with open(state.datapath + state.dataset + '_id2word.pkl', 'r') as f:
        id2word = cPickle.load(f)

    # load concatenate words
    print >> sys.stderr, 'loading concatenate words...'
    entity_words = np.load(state.datapath + state.dataset + '_concat-words.npz')[
        'entity_words'].astype('int32')
    M, N = entity_words.shape
    entity_words = entity_words.reshape((M, 1, N))
    print >> sys.stderr, 'entity_words.shape:', entity_words.shape

    # load word_embeddings
    word_embeddings = np.load(state.datapath + state.dataset + '-word-embeddings.npz')[
        'word_embeddings'].astype('float32')

    # entity_inputs = np.load(state.datapath + state.dataset + '-concat-word-vectors.npz')[
    #     'entity_inputs'].astype(theano.config.floatX)
    # print >> sys.stderr, 'max(entity_inputs): %s, min(entity_inputs): %s, mean: %s' % (
    #     np.max(entity_inputs), np.min(entity_inputs), np.mean(entity_inputs))
    # entity_inputs -= np.mean(entity_inputs)
    # entity_inputs *= 10
    # M, N = entity_inputs.shape
    # entity_inputs = entity_inputs.reshape((M, 1, 1, N))
    # print 'entity_inputs.shape:', entity_inputs.shape

    # Positives
    trainl = load_file(state.datapath + state.dataset + '-train-lhs.pkl')[:state.Nsyn, :]
    trainr = load_file(state.datapath + state.dataset + '-train-rhs.pkl')[:state.Nsyn, :]
    traino = load_file(state.datapath + state.dataset + '-train-rel.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        traino = traino[-state.Nrel:, :]

    # Valid set
    validl = load_file(state.datapath + state.dataset + '-valid-lhs.pkl')[:state.Nsyn, :]
    validr = load_file(state.datapath + state.dataset + '-valid-rhs.pkl')[:state.Nsyn, :]
    valido = load_file(state.datapath + state.dataset + '-valid-rel.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        valido = valido[-state.Nrel:, :]

    # Test set
    testl = load_file(state.datapath + state.dataset + '-test-lhs.pkl')[:state.Nsyn, :]
    testr = load_file(state.datapath + state.dataset + '-test-rhs.pkl')[:state.Nsyn, :]
    testo = load_file(state.datapath + state.dataset + '-test-rel.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        testo = testo[-state.Nrel:, :]

    batchsize = trainl.shape[1] / state.nbatches
    eval_batchsize = state.eval_batchsize
    entity_batchsize = state.entity_batchsize
    n_entity_batches = state.Nsyn / entity_batchsize

    print 'trainl.shape:', trainl.shape
    print 'trainr.shape:', trainr.shape
    print 'traino.shape:', traino.shape

    # Index conversion
    trainlidx, trainlidx_eval = toidx(trainl, state.neval)
    print 'trainlidx.shape:', trainlidx.shape
    print 'trainlidx_eval.shape:', trainlidx_eval.shape
    trainridx, trainridx_eval = toidx(trainr, state.neval)
    trainoidx, trainoidx_eval = toidx(traino, state.neval)
    validlidx, validlidx_eval = toidx(validl, state.neval)
    validridx, validridx_eval = toidx(validr, state.neval)
    validoidx, validoidx_eval = toidx(valido, state.neval)
    testlidx, testlidx_eval = toidx(testl, state.neval)
    testridx, testridx_eval = toidx(testr, state.neval)
    testoidx, testoidx_eval = toidx(testo, state.neval)

    # Model declaration
    if not state.loadmodel:
        # operators
        if state.op == 'Unstructured':
            leftop = Unstructured()
            rightop = Unstructured()
        elif state.op == 'SME_lin':
            leftop = LayerLinear(np.random, 'lin', state.ndim, state.ndim, state.nhid, 'left')
            rightop = LayerLinear(np.random, 'lin', state.ndim, state.ndim, state.nhid, 'right')
        elif state.op == 'SME_bil':
            leftop = LayerBilinear(np.random, 'lin', state.ndim, state.ndim, state.nhid, 'left')
            rightop = LayerBilinear(np.random, 'lin', state.ndim, state.ndim, state.nhid, 'right')
        elif state.op == 'SE':
            leftop = LayerMat('lin', state.ndim, state.nhid)
            rightop = LayerMat('lin', state.ndim, state.nhid)
        elif state.op == 'TransE':
            leftop = LayerTrans()
            rightop = Unstructured()
        # embeddings
        if not state.loademb:
            embeddings = Embeddings(np.random, state.Nent, state.ndim, 'emb')
        else:
            f = open(state.loademb)
            embeddings = cPickle.load(f)
            f.close()
        if state.op == 'SE' and type(embeddings) is not list:
            relationl = Embeddings(np.random, state.Nrel, state.ndim * state.nhid, 'rell')
            relationr = Embeddings(np.random, state.Nrel, state.ndim * state.nhid, 'relr')
            embeddings = [embeddings, relationl, relationr]
        if state.op == 'TransE' and type(embeddings) is not list:
            relationVec = Embeddings(np.random, state.Nrel, state.ndim, 'relvec')
            embeddings = [embeddings, relationVec, relationVec]
        simfn = eval(state.simfn + 'sim')
    else:
        f = open(state.loadmodel)
        embeddings = cPickle.load(f)
        leftop = cPickle.load(f)
        rightop = cPickle.load(f)
        simfn = cPickle.load(f)
        f.close()

    # Function compilation
    sem_model = build_model(entity_words.shape[2], state.ndim, word_embeddings, batch_size=batchsize)
    trainfunc = TrainSemantic(simfn, sem_model, embeddings, leftop,
                              rightop, margin=state.margin, rel=False)
    sem_func = SemanticFunc(sem_model)
    batch_ranklfunc = BatchRankLeftFnIdx(simfn, embeddings, leftop, rightop)
    batch_rankrfunc = BatchRankRightFnIdx(simfn, embeddings, leftop, rightop)

    out = []
    outb = []
    state.bestvalid = -1
    # relation_update_ratio = []

    print >> sys.stderr, "BEGIN TRAINING"
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]
        trainlidx = trainlidx[order]
        trainridx = trainridx[order]
        trainoidx = trainoidx[order]

        # Negatives
        trainln = create_random_mat(trainl.shape, np.arange(state.Nsyn))
        trainrn = create_random_mat(trainr.shape, np.arange(state.Nsyn))
        trainlnidx = convert2idx(trainln)
        trainrnidx = convert2idx(trainrn)

        # lhs_norms = []

        for i in range(state.nbatches):
            tmpo = traino[:, i * batchsize:(i + 1) * batchsize]
            tmpl_idx = trainlidx[i * batchsize:(i + 1) * batchsize]
            tmpr_idx = trainridx[i * batchsize:(i + 1) * batchsize]
            tmpnl_idx = trainlnidx[i * batchsize:(i + 1) * batchsize]
            tmpnr_idx = trainrnidx[i * batchsize:(i + 1) * batchsize]
            sem_inputl = entity_words[tmpl_idx]
            sem_inputr = entity_words[tmpr_idx]
            sem_inputnl = entity_words[tmpnl_idx]
            sem_inputnr = entity_words[tmpnr_idx]

            # print 'sem_inputl.shape:', sem_inputl.shape

            # training iteration
            outtmp = trainfunc(state.lrweights, state.momentum, state.lremb,
                               state.lrparam,
                               sem_inputl, sem_inputr, tmpo, sem_inputnl, sem_inputnr)
            # [cost, lhs, rhs, lhsn, rhsn, rell, relr, relation_updates] = outtmp[2:]
            # lhs_emb = outtmp[2]
            # relation_updates = np.array(outtmp[2])
            # relation_update_ratio.append(np.linalg.norm(relation_updates) / np.linalg.norm(embeddings[1].E.get_value()))
            # lhs = outtmp[2]
            # lhs_norm = np.mean([np.linalg.norm(j) for j in lhs])
            # lhs_norms.append(lhs_norm)
            out.append(outtmp[0])
            outb.append(outtmp[1])
            # embeddings normalization
            # if type(embeddings) is list:
            #     embeddings[0].normalize()
            # else:
            #     embeddings.normalize()

            # relation normalization
            relationVec.normalize()

            if i > 0 and i % state.printbatches == 0:
                print >> sys.stderr, 'batch %d.%d, cost: %f' % (
                    epoch_count, i, out[-1])
                # print >> sys.stderr, 'lhs norm: %f' % np.mean(lhs_norms)
                # lhs_norms = []

        print >> sys.stderr, 'Epoch %d, cost: %f' % (
            epoch_count, np.mean(out[-state.nbatches:]))
        # print >> sys.stderr, 'relation update ratio: %f' % np.mean(relation_update_ratio)
        # relation_update_ratio = []

        if (epoch_count % state.test_all) == 0:
            # model evaluation
            print >> sys.stderr, "-- EPOCH %s (%s seconds per epoch):" % (
                epoch_count,
                round(time.time() - timeref, 3) / float(state.test_all))
            timeref = time.time()
            print >> sys.stderr, "COST >> %s +/- %s, %% updates: %s%%" % (
                round(np.mean(out), 4), round(np.std(out), 4),
                round(np.mean(outb) * 100, 3))
            out = []
            outb = []

            # get sem output for all entities
            entity_embeddings = []
            for i in xrange(n_entity_batches):
                entity_embeddings.append(
                    sem_func(entity_words[i * entity_batchsize:(i + 1) * entity_batchsize])[0]
                )
            if n_entity_batches * entity_batchsize < state.Nsyn:
                entity_embeddings.append(
                    sem_func(entity_words[n_entity_batches * entity_batchsize:])[0]
                )
            entity_embeddings = np.vstack(entity_embeddings)

            resvalid = FastRankingScoreIdx(batch_ranklfunc, batch_rankrfunc, entity_embeddings,
                                           validlidx_eval, validridx_eval, validoidx_eval, eval_batchsize)
            state.valid = np.mean(resvalid[0] + resvalid[1])
            restrain = FastRankingScoreIdx(batch_ranklfunc, batch_rankrfunc, entity_embeddings,
                                           trainlidx_eval, trainridx_eval, trainoidx_eval, eval_batchsize)
            state.train = np.mean(restrain[0] + restrain[1])
            print >> sys.stderr, "\tMEAN RANK >> valid: %s, train: %s" % (
                state.valid, state.train)
            if state.bestvalid == -1 or state.valid < state.bestvalid:
                restest = FastRankingScoreIdx(batch_ranklfunc, batch_rankrfunc,
                                              entity_embeddings, testlidx_eval, testridx_eval,
                                              testoidx_eval, eval_batchsize)
                state.bestvalid = state.valid
                state.besttrain = state.train
                state.besttest = np.mean(restest[0] + restest[1])
                state.bestepoch = epoch_count
                # Save model best valid model
                f = open(state.savepath + '/best_valid_model.pkl', 'w')
                cPickle.dump(sem_model, f, -1)
                cPickle.dump(embeddings, f, -1)
                cPickle.dump(entity_embeddings, f, -1)
                cPickle.dump(leftop, f, -1)
                cPickle.dump(rightop, f, -1)
                cPickle.dump(simfn, f, -1)
                f.close()
                print >> sys.stderr, "\t\t##### NEW BEST VALID >> test: %s" % (
                    state.besttest)
            # Save current model
            f = open(state.savepath + '/current_model.pkl', 'w')
            cPickle.dump(sem_model, f, -1)
            cPickle.dump(embeddings, f, -1)
            cPickle.dump(entity_embeddings, f, -1)
            cPickle.dump(leftop, f, -1)
            cPickle.dump(rightop, f, -1)
            cPickle.dump(simfn, f, -1)
            f.close()
            state.nbepochs = epoch_count
            print >> sys.stderr, "\t(the evaluation took %s seconds)" % (
                round(time.time() - timeref, 3))
            timeref = time.time()
            channel.save()
    return channel.COMPLETE


def launch(datapath='data/', dataset='FB15k', Nent=16296,
           Nsyn=14951, Nrel=1345, loadmodel=False, loademb=False, op='Unstructured',
           simfn='Dot', ndim=50, nhid=50, margin=1., lrweights=0.1, momentum=0.9,
           lremb=0.1, lrparam=1., nbatches=4000, totepochs=2000, test_all=1, neval=50,
           seed=123, savepath='.', eval_batchsize=500000, entity_batchsize=20000,
           printbatches=1):
    # Argument of the experiment script
    state = DD()

    state.datapath = datapath
    state.dataset = dataset
    state.Nent = Nent
    state.Nsyn = Nsyn
    state.Nrel = Nrel
    state.loadmodel = loadmodel
    state.loademb = loademb
    state.op = op
    state.simfn = simfn
    state.ndim = ndim
    state.nhid = nhid
    state.margin = margin
    state.lrweights = lrweights
    state.momentum = momentum
    state.lremb = lremb
    state.lrparam = lrparam
    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all
    state.neval = neval
    state.seed = seed
    state.savepath = savepath
    state.eval_batchsize = eval_batchsize
    state.entity_batchsize = entity_batchsize
    state.printbatches = printbatches

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    # Jobman channel remplacement
    class Channel(object):
        def __init__(self, state):
            self.state = state
            f = open(self.state.savepath + '/orig_state.pkl', 'w')
            cPickle.dump(self.state, f, -1)
            f.close()
            self.COMPLETE = 1

        def save(self):
            f = open(self.state.savepath + '/current_state.pkl', 'w')
            cPickle.dump(self.state, f, -1)
            f.close()

    channel = Channel(state)

    FB15kexp(state, channel)


if __name__ == '__main__':
    launch()
