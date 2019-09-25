from multiprocessing import Pool
import math, random, sys
import cPickle as pickle
import argparse
from functools import partial
import torch

from hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit

def tensorize(mol_batch, vocab):
    return MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)[:-1] #no need of order for x
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return x + y

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = torch.tensor(cond).int()
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)[:-1] #no need of order for x
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return x + y + (cond,)

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='pair')
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
    args.vocab = PairVocab(vocab, cuda=False)

    pool = Pool(args.ncpu)
    random.seed(1)

    if args.mode == 'pair':
        #dataset contains molecule pairs
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:2] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in xrange(0, len(data), args.batch_size)]
        func = partial(tensorize_pair, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) / 1000, 1)

        le = (len(all_data) + num_splits - 1) / num_splits

        for split_id in xrange(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'cond_pair':
        #dataset contains molecule pairs with conditions
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:3] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in xrange(0, len(data), args.batch_size)]
        func = partial(tensorize_cond, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) / 1000, 1)

        le = (len(all_data) + num_splits - 1) / num_splits

        for split_id in xrange(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'single':
        #dataset contains single molecules
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[0] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in xrange(0, len(data), args.batch_size)]
        func = partial(tensorize, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = len(all_data) / 1000

        le = (len(all_data) + num_splits - 1) / num_splits

        for split_id in xrange(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

