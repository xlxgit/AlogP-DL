# -*- coding: utf-8 -*-#
'''
# Name:         Regression-ensemble
'''

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, concatenate, Dropout, Flatten
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from pandas import read_csv
from keras.regularizers import l2
from keras.constraints import maxnorm
import pickle
import scipy
from scipy import stats
import tensorflow as tf
from keras import backend as K

model_id="Regression-learning-approved-joined"


def load_data():
    dataframe1 = read_csv("../ecfp-drugbank-approved.csv", delimiter = ',')
    dataframe2 = read_csv("../maccskeys-drugbank-approved.csv", delimiter = ',')

    dataset1 = dataframe1.values
    dataset2 = dataframe2.values

    # split into input (X) and output (Y) variables
    X1 = dataset1[:,:]
    X2 = dataset2[:,:]
    Y = np.loadtxt("../drugbank_approved_logP.AlogP.value")

    x1_train, x1_test, y1_train, y1_test = train_test_split(X1, Y, test_size=0.2, random_state=42)
    x2_train, x2_test, y2_train, y2_test = train_test_split(X2, Y, test_size=0.2, random_state=42)
    
    return (x1_train, y1_train), (x1_test, y1_test), (x2_train, y2_train), (x2_test, y2_test)


def draw_train_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def build_model():
    inputs1 = Input(shape=(2048, ))
    inputs2 = Input(shape=(167, ))
    model1_1 = Dense(1000, input_dim=2048, kernel_initializer='normal', activation='relu', kernel_regularizer=l2(0.0003), W_constraint=maxnorm(4))(inputs1)
    model1_2 = Dense(50, kernel_initializer='normal', activation='relu',kernel_regularizer=l2(0.0003), W_constraint=maxnorm(4))(model1_1)
    model2_1 = Dense(200, input_dim=167, kernel_initializer='normal', activation='relu', kernel_regularizer=l2(0.0003), W_constraint=maxnorm(4))(inputs2)
    model2_2 = Dense(300, kernel_initializer='normal', activation='relu',kernel_regularizer=l2(0.0003), W_constraint=maxnorm(4))(model2_1)
    model1_3 = Dropout(0.6)(model1_2)
    model2_3 = Dropout(0.2)(model2_2)
    #con = concatenate([model1_4, model2_4])
    #output = Dense(1, activation='relu')(con1)
    output1 = Dense(1, activation='linear')(model1_3)
    output2 = Dense(1, activation='linear')(model2_3)
    model = Model(inputs=[inputs1, inputs2], outputs=[output1, output2])
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=["accuracy"],
                  #loss_weights=[0.5, 0.5]
                  )
    return model


if __name__ == '__main__':
    (x1_train, y1_train), (x1_test, y1_test), (x2_train, y2_train), (x2_test, y2_test)  = load_data()

    for i in range(20):
        
        K.clear_session()                                                                    
        tf.reset_default_graph()
        
        model = build_model()
        early_stopping = EarlyStopping(monitor='loss', patience=10)
        history = model.fit([x1_train, x2_train], [y1_train, y2_train],
                epochs=100,
                batch_size=40,
                validation_split=0.2,
                callbacks=[early_stopping])
	#draw_train_history(history)
	#model.save("regression-learning-ensemble.h5")
        logFilePath = './log'+str(model_id)+'.txt'
        fobj = open(logFilePath, 'a')
        fobj.write('model id: ' + str(model_id)+'\n')
        fobj.write('x1_train shape: ' + str(x1_train.shape) + '\n')
        fobj.write('x1_test shape: ' + str(x1_test.shape)+'\n')
        fobj.write('x2_train shape: ' + str(x2_train.shape) + '\n')
        fobj.write('x2_test shape: ' + str(x2_test.shape)+'\n')
        #fobj.write('training accuracy: ' + str(history.history['acc'][-1]) + '\n')
        fobj.write('training loss: ' + str(history.history['loss'][-1]) + '\n')
        fobj.write('---------------------------------------------------------------------------\n')
        fobj.write('\n')
        fobj.close()
	    
	#model.fit([x1_train, x2_train], [y1_train, y2_train])
        predicted_train=model.predict([x1_train, x2_train])
        predicted_traintxt="predicted_train"+str(model_id)+'.txt'
        true_traintxt     ="true_train"+str(model_id)+'.txt'
        np.savetxt(predicted_traintxt, predicted_train, fmt='%f')
        np.savetxt(true_traintxt, y1_train, fmt='%f')
        predicted0=np.loadtxt(predicted_traintxt)
        y_train=np.loadtxt(true_traintxt)
        avg0=np.mean(predicted0, axis=0)
        avg0txt="predicted_train_avg_"+str(model_id)+'.txt'
        np.savetxt(avg0txt, avg0, fmt='%f')
        print("train_pearson0: ",scipy.stats.pearsonr(y1_train,predicted0[0]))
        print("train_pearson1: ",scipy.stats.pearsonr(y1_train,predicted0[1]))
        print("train_pearson_avg: ",scipy.stats.pearsonr(y1_train,avg0))
	
        predicted_test=model.predict([x1_test, x2_test])
        predicted_testtxt="predicted_test"+str(model_id)+'.txt'
        true_testtxt     ="true_test"+str(model_id)+'.txt'
        np.savetxt(predicted_testtxt, predicted_test, fmt='%f')
        np.savetxt(true_testtxt, y1_test, fmt='%f')
        predicted1=np.loadtxt(predicted_testtxt)
        y_test=np.loadtxt(true_testtxt)
        avg1=np.mean(predicted1, axis=0)
        avg1txt="predicted_test_avg_"+str(model_id)+'.txt'
        np.savetxt(avg1txt, avg1, fmt='%f')
        print("test_pearson0: ",scipy.stats.pearsonr(y1_test,predicted1[0]))
        print("test_pearson1: ",scipy.stats.pearsonr(y1_test,predicted1[1]))
        print("test_pearson_avg: ",scipy.stats.pearsonr(y1_test,avg1))
	
	
        loss = model.evaluate([x1_train, x2_train], [y1_train, y2_train], batch_size=40)
        print("train loss: {}".format(loss))
        loss = model.evaluate([x1_train, x2_train], [y1_train, y2_train], batch_size=40)
        print("test loss: {}".format(loss))

file = open('./history'+str(model_id)+'.pkl', 'wb')
pickle.dump(history.history, file)
file.close()
