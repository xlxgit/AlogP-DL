import scipy
from scipy import stats
import numpy as np
Y_test=np.loadtxt("true_testRegression-learning-approved-joined.txt")
prediction=np.loadtxt("predicted_test_avg_Regression-learning-approved-joined.txt")
print("RF-maccs person: ",scipy.stats.pearsonr(Y_test,prediction))

from sklearn.metrics import r2_score
print("r2_score: ",r2_score(Y_test,prediction))

