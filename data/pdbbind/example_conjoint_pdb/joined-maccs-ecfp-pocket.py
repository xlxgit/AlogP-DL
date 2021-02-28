from pandas import read_csv
import numpy as np
import pandas as pd

# load dataset
dataframe1 = read_csv("maccs-pocket-refined2019.txt", delimiter = ',', header=None)
dataframe2 = read_csv("ecfp-pocket-refined2019.txt", delimiter = ',', header=None)

dataframe=pd.concat([dataframe1, dataframe2], axis=1)

dataset = dataframe.values
X = dataset[:,:]

print (len(X),)
a=np.concatenate((X[...,0:333],X[...,334:]),axis=1)
np.savetxt('conjoint-maccs-ecfp-pocket.csv',a, fmt='%s', delimiter=',')
exit()

