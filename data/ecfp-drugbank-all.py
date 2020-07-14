
import pandas as pd
import numpy as np
from rdkit import Chem
from deepchem.feat import fingerprints as fp

#suppl2 = Chem.SDMolSupplier('/home/xlxhku/data/TCM/Struct/DrugBank/structures_drugbank_approved.sdf')
suppl2 = Chem.SDMolSupplier('drugbank_approved_logP.sdf')
suppl2 = Chem.SDMolSupplier('drugbank_all_3d_logP.sdf')

size=2048
ID_list = []
error_ID_list = []
feature_list = []
for mol in suppl2:
    mol = [mol]  #如果不加此行，则TypeError: 'Mol' object is not iterable
    engine = fp.CircularFingerprint(radius=2, size=2048, chiral=False,
            bonds=True,features=False, sparse=False, smiles=False) 
    feature = engine(mol) #结果形式为：[array([0, 0, 0, ..., 0, 0, 0]
    feature_list.extend(feature) #如果用append，则为[[array([0, 0, 0, ..., 0, 0, 0])],……]]型，生成dataframe时出现ValueError: Must pass 2-d input
                             
ID_feature_df = pd.DataFrame(feature_list)
vec_name = ['feature_{0}'.format(i) for i in range(0,size)]
#ID_feature_df.columns = vec_name
#ID_feature_df.index.name = 'compound_ID'
                              
                               
ID_feature_df.to_csv('ecfp-drugbank-all.csv',index=0)
