#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

from sklearn import svm
from sklearn.metrics import accuracy_score
#m=svm.SVC(kernel='linear')
#m = svm.SVC(kernel='rbf')
#m = svm.SVC(kernel='rbf',C=10)
#m = svm.SVC(kernel='rbf',C=100)
#m = svm.SVC(kernel='rbf',C=1000)
m = svm.SVC(kernel='rbf',C=10000)
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
t0 = time()
m.fit(features_train,labels_train)
print "Training time:" , round(time()-t0,3) , "s"
t1 = time()
pred=m.predict(features_test)
print "Predict time:", round(time()-t1,3) , "s"
print accuracy_score(labels_test,pred)
#print pred[10]
#print pred[26]
#print pred[50]
i = 0
for a in pred:
	if a==1:
		i=i+1
print i
