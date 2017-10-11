#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter
import numpy as np
import math
from collections import Counter


class MyNearestNeighborClassifier(object):
    
    X_train = []
    y_train = []
    predictions = []
    
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    
    # Helper functions
    # given two data points, calculate the euclidean distance between them
    def get_distance(self, data1, data2):
        points = zip(data1, data2)
        diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
        return math.sqrt(sum(diffs_squared_distance))

    def get_tuple_distance(self, train, test_instance):
        return (train, self.get_distance(test_instance, train[0]))
    
    # given a training set and a test instance, use getDistance to calculate all pairwise distances
    # returns sorted distances between a test case and all training cases is needed. 
    def get_neighbours(self, training_set, test_instance):
        distances = [self.get_tuple_distance(train, test_instance) for train in training_set]

        # index 1 is the calculated distance between train and test_instance
        sorted_distances = sorted(distances, key=itemgetter(1))

        # extract only training instances
        sorted_training_instances = [tuple[0] for tuple in sorted_distances]

        # select first k elements
        return sorted_training_instances[:self.n_neighbors]

    # 3) given an array of nearest neighbours for a test case, tally up their classes to vote on test case class
    def get_majority_vote(self, neighbours):
        # index 1 is the class
        classes = [neighbour[1] for neighbour in neighbours]
        count = Counter(classes)
        return count.most_common()[0][0] 
    
    # Main functions
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
            
    def predict(self, X_test, y_test):
        # reformat datasets for convenience, zip(): returns an iterator of tuples.
        train = np.array(list(zip(self.X_train, self.y_train)))
        test = np.array(list(zip(X_test, y_test)))

        # generate predictions
        # for each instance in the test set, get nearest neighbours and majority vote on predicted class
        for x in range(len(X_test)):
            neighbours = self.get_neighbours(train, test_instance=test[x][0])
            majority_vote = self.get_majority_vote(neighbours)
            self.predictions.append(majority_vote)
        return self.predictions

    
# Preparing test and train data using Iris
#----------------------
# load the data and create the training and test sets
# random_state = 1 is just a seed to permit reproducibility of the train/test split
iris = load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y) #.unique finds unique elements in array

# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
X_train = iris_X[indices[:-10]]
y_train = iris_y[indices[:-10]]
X_test  = iris_X[indices[-10:]]
y_test  = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)

print ('Overall accuracy of the Sckit Model is: ' + '{:f}'.format(accuracy_score(y_test, y_pred)))

# Testing the KNN
#----------------------
clf = MyNearestNeighborClassifier(n_neighbors=3)

# storing the training data and labels.
clf.fit(X_train, y_train)

#return the predicted class index for each example from the test set, just like the built-in method.
y_pred = clf.predict(X_test, y_test)
print ('Overall accuracy of My Model is: ' + '{:f}'.format(accuracy_score(y_test, y_pred)))