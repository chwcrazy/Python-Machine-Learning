#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################

########## Naive Bayes ##########
###								#
###								#
######################################################################################################
t0 = time()
### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.naive_bayes import GaussianNB

### create classifier
clf = GaussianNB()#TODO

### fit the classifier on the training features and labels
#TODO
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)#TODO
print "predict time:", round(time()-t1, 3), "s"

 # Method 1
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy
print "total time:", round(time()-t0, 3), "s"


#############  SVM  #############
###								#
###								#
######################################################################################################
from sklearn import svm

t0 = time()
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
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy


######### Decision Tree #########
###								#
###								#
######################################################################################################
from sklearn import tree
from sklearn.metrics import accuracy_score
t0 = time()
clf = tree.DecisionTreeClassifier()

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"

# acc_min_samples_split_2 = accuracy_score(labels_test, pred)


# Method 1
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy

####### K Nearest Neighbors #####
###								#
###								#
######################################################################################################
from sklearn.neighbors import KNeighborsClassifier
t0 = time()

clf = KNeighborsClassifier(n_neighbors=1) # default value is 2
# clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(features_train)
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

distances, indices = clf.kneighbors(features_train)

t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"

# # Method 1
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy

############ AdaBoost ###########
###								#
###								#
######################################################################################################
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

t0 = time()
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"

# # # Method 1
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy




########## Random Forest ########
###								#
###								#
######################################################################################################

from sklearn.ensemble import RandomForestClassifier

t0 = time()
clf = RandomForestClassifier(n_estimators=10)
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"

# # # Method 1
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy


######################################################################################################

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass


#########################       Result      ##############################


# Naive Bayes --> accuracy: 0.884
# SVM --> accuracy: 0.932
# Decision Tree --> accuracy: 0.908
# K Nearest Neighbors --> accuracy: 0.94
# AdaBoost --> accuracy: 0.924
# Random Forest --> accuracy: 0.916


