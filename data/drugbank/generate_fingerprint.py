
import pandas as pd
import numpy as np
from rdkit import Chem
from deepchem.feat import fingerprints as fp
from rdkit import rdBase, Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Draw import SimilarityMaps
from deepchem.feat import fingerprints as fp

smiles_file=pd.read_csv('drugbank_approved_logP_smi.csv', sep=',',usecols=['logp', 'smiles'], chunksize=10)

maccs_size=167
ecfp_size=2048
MACCS_feature_list = []
ECFP_feature_list = []
logP_list = []
for chunk in smiles_file:
    for logP,smiles in zip(chunk['logp'],chunk['smiles']):
        mol = Chem.MolFromSmiles(smiles)

        maccs_feature = AllChem.GetMACCSKeysFingerprint(mol) 
        maccsfeaturebit = list(maccs_feature.ToBitString())
        MACCS_feature_list.append(maccsfeaturebit) 
        
        mol = [mol]
        engine = fp.CircularFingerprint(radius=2, size=2048, chiral=False,
                bonds=True,features=False, sparse=False, smiles=False) 
        ECFP_feature = engine(mol)
        logP_list.append(logP)
        ECFP_feature_list.extend(ECFP_feature) 

MACCS_feature_df = pd.DataFrame(MACCS_feature_list)
vec_name = ['feature_{0}'.format(i) for i in range(0,maccs_size)]
print("MACCS keys stored in maccskeys-approved.csv\n")
MACCS_feature_df.to_csv('maccskeys-approved.csv',index=0)

ECFP_feature_df = pd.DataFrame(ECFP_feature_list)
logP_df = pd.DataFrame(logP_list)
vec_name = ['feature_{0}'.format(i) for i in range(0,ecfp_size)]
ECFP_feature_df.columns = vec_name
print("ECFP stored in ecfp-approved.csv\n")
ECFP_feature_df.to_csv('ecfp-approved.csv',mode='a',index=False, header=None)

print("logP value stored in approved_logP.value\n")
logP_df.to_csv('approved_logP.value',mode='a',index=False, header=None)
