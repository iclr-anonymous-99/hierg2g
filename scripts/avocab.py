from rdkit import Chem
import sys

if __name__ == "__main__":
    vocab = set()
    for line in sys.stdin:
        s = line.strip("\r\n ")
        mol = Chem.MolFromSmiles(s)
        for atom in mol.GetAtoms():
            vocab.add( (atom.GetSymbol(), atom.GetFormalCharge()) )

    for x,y in vocab:
        print x,y
