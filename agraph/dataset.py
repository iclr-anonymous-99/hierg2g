import torch
from rdkit import Chem
from chemutils import get_leaves
from torch.utils.data import Dataset
from mol_graph import MolGraph
import os, random, gc
import cPickle as pickle

class MoleculeDataset(Dataset):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in xrange(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize(self.batches[idx], self.avocab)

class MolEnumRootDataset(Dataset):

    def __init__(self, data, avocab, num_decode):
        self.batches = data
        self.avocab = avocab
        self.num_decode = num_decode

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.batches[idx])
        leaves = get_leaves(mol)
        smiles_list = [Chem.MolToSmiles(mol, rootedAtAtom=i, isomericSmiles=False) for i in leaves]
        smiles_list = list( set(smiles_list) )
        while len(smiles_list) < self.num_decode:
            smiles_list = smiles_list + smiles_list
        smiles_list = smiles_list[:self.num_decode]
        return MolGraph.tensorize(smiles_list, self.avocab)

class MolPairDataset(Dataset):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in xrange(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y = zip(*self.batches[idx])
        x = MolGraph.tensorize(x, self.avocab)[:-1] #no need of order for x
        y = MolGraph.tensorize(y, self.avocab)
        return x + y

class CondPairDataset(Dataset):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in xrange(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y, cond = zip(*self.batches[idx])
        cond = [map(int, c.split(',')) for c in cond]
        cond = torch.tensor(cond).float()
        x = MolGraph.tensorize(x, self.avocab)[:-1] #no need of order for x
        y = MolGraph.tensorize(y, self.avocab)
        return x + y + (cond,)

class DataFolder(object):

    def __init__(self, data_folder, batch_size, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                batches = pickle.load(f)

            if self.shuffle: random.shuffle(batches) #shuffle data before batch
            for batch in batches:
                yield batch

            del batches
            gc.collect()

