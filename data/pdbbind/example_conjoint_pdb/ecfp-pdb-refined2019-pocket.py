
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


ligand = Chem.MolFromPDBFile('11gs_ligand.pdb')
pocketr1 = Chem.MolFromPDBFile('11gs_pocket_clean.pdb')

featureL=AllChem.GetMorganFingerprintAsBitVect(ligand,2,nBits = 1024)
featureL1=AllChem.GetMorganFingerprintAsBitVect(pocketr1, 2,nBits = 1024)

features=[]
features=['11gs']
features.extend(featureL.ToBitString())
features.extend(featureL1.ToBitString())

with open('ecfp-pocket-refined2019.txt', 'a') as f:
    f.write(','.join([str(x) for x in features]))
    f.write('\n')
exit()
