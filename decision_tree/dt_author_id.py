#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
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

print len(features_train[0])

from sklearn import tree
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier(min_samples_split=40)

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"

acc_min_samples_split_2 = accuracy_score(labels_test, pred)


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
#########################################################


