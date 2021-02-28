
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

ligand = Chem.MolFromPDBFile('11gs_ligand.pdb')
pocketr1 = Chem.MolFromPDBFile('11gs_pocket_clean.pdb')
featureL=MACCSkeys.GenMACCSKeys(ligand)
featureL1=MACCSkeys.GenMACCSKeys(pocketr1)
features=[]
features=['11gs']
features.extend(featureL.ToBitString()[1:167])
features.extend(featureL1.ToBitString()[1:167])
with open('maccs-pocket-refined2019.txt', 'a') as f:
    f.write(','.join([str(x) for x in features]))
    f.write('\n')
exit()
