#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd
import csv
import itertools
from sklearn.metrics import classification_report, precision_recall_curve

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go
import collections
import os
# from pydotplus import graphviz


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def preProcess():
    df = pd.read_csv('crabs.csv')

    #the accuracy doenst change if the column index is not droped
    df.drop(['index'], 1, inplace = True)
    # presentHist(df)

    #define x and y
    #change the data in order to be processed
    y = np.array(df['sp'])
    df['sex'].replace('M', 1,inplace=True)
    df['sex'].replace('F', 0,inplace=True)

    df['sp'].replace('O', 0,inplace=True)
    df['sp'].replace('B', 3,inplace=True)
    X = np.array(df.drop(['sp'],1))
    return [X,y]

if __name__ == '__main__':
    min_LeafSamples, min_SplitSamples = int(sys.argv[1]), int(sys.argv[2])

    tree = pd.read_csv('crabs.csv',sep= ',', header= None)

    lis = preProcess()
    X, Y= lis[0],lis[1]

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

    #determine the min_samples_leaf and min_samples_split
    clf_gini = DecisionTreeClassifier( min_samples_leaf=min_LeafSamples,min_samples_split=min_SplitSamples)
    clf = clf_gini.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    print('k-NN score for test set: %f' % clf.score(X_test, y_test))
    print('k-NN score for training set: %f' % clf.score(X_train, y_train))
    print(classification_report(y_true, y_pred))

    #print decision confusion_matrix
    y_score_n = clf.predict(X_test)
    conf_matrix = (confusion_matrix(y_test, y_score_n))
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=["B","O"],title='Confusion matrix, without normalization')



    # from IPython.display import Image

    dot_data = export_graphviz(clf,out_file=("tree.dot"),feature_names=["sex","FL","RW","CL","CW","BD"],class_names=["B","O"] ,filled= True,rounded=True)

    #generate the tree in a png file
    os.system("dot -Tpng tree.dot -o tree.png")







#
