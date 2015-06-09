#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

t0 = time()
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Smaller training set  1% 
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
from sklearn import svm
# kernel: linear
# linear_svc = svm.SVC(kernel='linear')
# kernel: rbf
rbf_svc = svm.SVC(kernel='rbf', C=10000.0)

### fit the classifier on the training features and labels
# linear_svc.fit(features_train, labels_train)
rbf_svc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features
t1 = time()
# pred = linear_svc.predict(features_test)
pred = rbf_svc.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"

# Method 1
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(labels_test, pred)
# print accuracy
# Method 2
correct_amt = 0.0
output = []
for i in range(len(pred)):
    if pred[i] == labels_test[i]:
        correct_amt += 1
    else:
    	output.append((str(pred[i]), str(labels_test[i])))
        # print str(pred[i]) + " " + str(labels_test[i])
if len(pred) == 0:
    accuracy = 1
else:
    accuracy = correct_amt / len(pred)
print accuracy
res = 0
for i in range(len(pred)):
	if pred[i] == 1:
		res += 1
print 'res', res
print len(pred)
#########################################################

# Different C parameter in %1 Data Set
# 10.0 : 0.616040955631
# 100.0 : 0.616040955631
# 1000.0 : 0.821387940842
# 10000.0 : 0.892491467577
