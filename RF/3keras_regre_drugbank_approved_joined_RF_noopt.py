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

from sklearn.ensemble import RandomForestRegressor

model_id = "drugbank_approved_logP_RF_joined_noopt"
#dataframe = read_csv("../ecfp-drugbank-approved.csv", delimiter = ',')
#dataframe = read_csv("../maccskeys-drugbank-approved.csv", delimiter = ',')
dataframe1 = read_csv("../ecfp-drugbank-approved.csv", delimiter = ',')
dataframe2 = read_csv("../maccskeys-drugbank-approved.csv", delimiter = ',')
dataframe  = pd.concat([dataframe1, dataframe2], axis=1)

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,:]
Y = np.loadtxt("../drugbank_approved_logP.AlogP.value")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# define wider model
#model = RandomForestRegressor(n_estimators=40, max_depth=12, min_samples_split=2, min_samples_leaf=5, max_features='sqrt', oob_score=True,)

import pickle
import scipy
from scipy import stats
nlayer=1
for i in range(20):
    estimators = RandomForestRegressor(n_estimators=40, max_depth=9, min_samples_split=10, min_samples_leaf=5, 
            max_features='sqrt', oob_score=True, )

    history=estimators.fit(X_train, Y_train, )
    print(history.oob_score_)
    logFilePath = './log'+str(model_id)+'.txt'
    fobj = open(logFilePath, 'a')
    fobj.write('model id: ' + str(model_id)+'\n')
    fobj.write('x_train shape: ' + str(X_train.shape) + '\n')
    fobj.write('x_test shape: ' + str(X_test.shape)+'\n')
    fobj.write('training accuracy: ' + str(history.oob_score_) + '\n')
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

