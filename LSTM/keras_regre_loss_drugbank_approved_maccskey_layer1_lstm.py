from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l2
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
import pandas as pd
import tensorflow as tf
from keras import backend as K

# load dataset
model_id = "drugbank_approved_logP_maccskeys_layer1_LSTM"
#dataframe = read_csv("ecfp-drugbank-approved.csv", delimiter = ',')
dataframe = read_csv("../maccskeys-drugbank-approved.csv", delimiter = ',')

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,:]
Y = np.loadtxt("../drugbank_approved_logP.AlogP.value")
#Y = dataset[:,13]
print (len(X),len(Y))
print(X[1],Y[1])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
data_dim = [2011,167]
n_timesteps=1
# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(LSTM(200, input_dim=167,return_sequences=True,input_shape=(n_timesteps,data_dim) ))
    model.add(LSTM(300,  ))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear', kernel_initializer='normal'))
    # Compile model
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=["accuracy"])
    return model

# evaluate model with standardized dataset

import pickle
import scipy
from scipy import stats
nlayer=1
for i in range(20):
    K.clear_session()                                                                    
    tf.reset_default_graph()
    
    estimators =  KerasRegressor(build_fn=wider_model,  epochs=100, batch_size=20, verbose=0)
    history=estimators.fit(X_train, Y_train, epochs=100, batch_size=20,  validation_split=0.2)
    print("loss:", history.history['loss'])
    print("acc:",  history.history['acc'])
    logFilePath = './log'+str(model_id)+'.txt'
    fobj = open(logFilePath, 'a')
    fobj.write('model id: ' + str(model_id)+'\n')
    fobj.write('x_train shape: ' + str(X_train.shape) + '\n')
    fobj.write('x_test shape: ' + str(X_test.shape)+'\n')
    fobj.write('training accuracy: ' + str(history.history['acc'][-1]) + '\n')
    fobj.write('training loss: ' + str(history.history['loss'][-1]) + '\n')
    #fobj.write('model evaluation results: ' + str(score[0]) + '  ' +str(score[-1])+'\n')
    fobj.write('---------------------------------------------------------------------------\n')
    fobj.write('\n')
    fobj.close()
        
    prediction0=estimators.predict(X_train)
    print("train-pearson: ",scipy.stats.pearsonr(Y_train,prediction0))
    prediction=estimators.predict(X_test)
    print("test-pearson: ",scipy.stats.pearsonr(Y_test,prediction))


file = open('./history'+str(model_id)+'.pkl', 'wb')
pickle.dump(history.history, file)
file.close()

predictedtxt = 'predicted_'+str(model_id)+'.txt'


#estimators.fit(X, Y)
prediction=estimators.predict(X_test)
#print(X, prediction, dataset[:,13])

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

fig = plt.figure()#新建一张图
plt.plot(history.history['acc'],label='training acc',)
#plt.plot(history.history['val_acc'],label='val acc',marker=">",c="gray")
plt.plot(history.history['val_acc'],label='val acc',)
plt.title('model accuracy', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.legend(loc='best', fontsize=12)
plt.tick_params(labelsize=15)
fig.savefig('A1'+str(model_id)+'acc.png')

fig = plt.figure()
plt.plot(history.history['loss'],label='training loss',)
plt.plot(history.history['val_loss'], label='val loss',)
plt.title('model loss', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.tick_params(labelsize=15)
plt.legend(loc='best', fontsize=12)
fig.savefig('A1'+str(model_id)+'loss.png', dpi=300)

prediction0=estimators.predict(X_train)
predictedtxt0 = 'predicted_train_'+str(model_id)+'.txt'
truetxt0 = 'true_train_'+str(model_id)+'.txt'
np.savetxt(predictedtxt0, prediction0, fmt='%f')
np.savetxt(truetxt0, Y_train, fmt='%f')

predictedtxt = 'predicted_'+str(model_id)+'.txt'
truetxt = 'true_'+str(model_id)+'.txt'
np.savetxt(predictedtxt, prediction, fmt='%f')
np.savetxt(truetxt, Y_test, fmt='%f')

fig = plt.figure()#新建一张图
plt.ylabel('predicted logP', fontsize=20)
plt.xlabel('true logP', fontsize=20)
plt.tick_params(labelsize=15)
plt.scatter( Y_test, prediction, s=50, alpha=0.8,)
#plt.show()
plt.savefig('A1'+str(model_id)+'predited.png', dpi=300)
