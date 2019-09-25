from hgraph import *
import sys
from rdkit import Chem

if __name__ == "__main__":
    vocab = set()
    for line in sys.stdin:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )

    for x,y in vocab:
        print x,y
