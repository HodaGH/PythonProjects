""" Multiclass classification using SVM """

from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss #If normalize is True, return the fraction of misclassifications (float), else it returns the number of misclassifications (int). The best performance is 0.

#import data 
data =pd.read_csv('/Users/Hoda/Desktop/Insight/wine/winequality-red.csv').values
X = data[:,0:10] # first evelen features 
Y = data[:, 11] # last column as target

"""
clf = SVC()
clf.fit(X,Y)
print(clf) # C= 1.0, decision_function_shape=None, degree =3, gamma='auto', kernel='rbf' ....
"""

C =1.0 # SVM regularization parameter
clf = SVC(kernel='linear', C=C)
clf.fit(X,Y)
y_pred = clf.predict(X)
classif_rate = np.mean(y_pred.ravel()==Y.ravel())*100 #numpy.ravel(a, order='C'). A 1-D array, containing the elements of the input, is returned
accuracy = zero_one_loss(Y.ravel(),y_pred.ravel())
print("classif_accuracy is: %f" %accuracy) 
print("classif_rate is: %f" % classif_rate) #error rate: fraction of missclassifications 






