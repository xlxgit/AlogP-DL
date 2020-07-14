from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from  sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
import pandas as pd
from sklearn import svm

model_id = "drugbank_approved_joined_logP_SVR_rbf"
dataframe1 = read_csv("../ecfp-drugbank-approved.csv", delimiter = ',')
dataframe2 = read_csv("../maccskeys-drugbank-approved.csv", delimiter = ',')
dataframe  = pd.concat([dataframe1, dataframe2], axis=1)

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,:]
Y = np.loadtxt("../drugbank_approved_logP.AlogP.value")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# define wider model
model =  svm.SVR(kernel='rbf', C=10, gamma=0.2)

#hyperparameter; do it in the following pair


C=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
kernel=['rbf', 'linear', 'poly','sigmoid']
gamma=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
#coef0 for poly and sigmoid; degree for poly
coef0=[]
degree=[1,3,5,7,9]
"""
example 
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                       coef0=1)
linear: C
poly: C, degree, gamma, coef0
rbf: C, gamma, 
sigmoid: C, gamma, coef0
"""
resultscsv='gridscores-'+str(model_id)+".csv"

import pickle
import scipy
from scipy import stats
nlayer=1
for i in range(1):
    estimators = svm.SVR(kernel='rbf',C=100, gamma=0.1 )

    history=estimators.fit(X_train, Y_train, )
    print(history.score(X_train, Y_train,))
    logFilePath = './log'+str(model_id)+'.txt'
    fobj = open(logFilePath, 'a')
    fobj.write('model id: ' + str(model_id)+'\n')
    fobj.write('x_train shape: ' + str(X_train.shape) + '\n')
    fobj.write('x_test shape: ' + str(X_test.shape)+'\n')
    fobj.write('training accuracy: ' + str(history.score(X_train, Y_train,)) + '\n')
    #fobj.write('model evaluation results: ' + str(score[0]) + '  ' +str(score[-1])+'\n')
    fobj.write('---------------------------------------------------------------------------\n')
    fobj.write('\n')
    fobj.close()
    
    prediction0=estimators.predict(X_train)
    print("train-pearson: ",scipy.stats.pearsonr(Y_train,prediction0))

    prediction=estimators.predict(X_test)
    print("test-pearson: ",scipy.stats.pearsonr(Y_test,prediction))
#hyperparameter; do it in the following pair

#save raw output
prediction0=estimators.predict(X_train)
predictedtxt0 = 'predicted_train_'+str(model_id)+'.txt'
truetxt0 = 'true_train_'+str(model_id)+'.txt'
np.savetxt(predictedtxt0, prediction0, fmt='%f')
np.savetxt(truetxt0, Y_train, fmt='%f')

predictedtxt = 'predicted_'+str(model_id)+'.txt'
truetxt = 'true_'+str(model_id)+'.txt'
np.savetxt(predictedtxt, prediction, fmt='%f')
np.savetxt(truetxt, Y_test, fmt='%f')

exit()
