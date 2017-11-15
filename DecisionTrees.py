#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd
import csv
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
import plotly.plotly as py
import plotly.graph_objs as go
import collections
import os
# from pydotplus import graphviz

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

    # from IPython.display import Image

    dot_data = export_graphviz(clf,out_file=("tree.dot"),feature_names=["sex","FL","RW","CL","CW","BD"],class_names=["B","O"] ,filled= True,rounded=True)

    #generate the tree in a png file
    os.system("dot -Tpng tree.dot -o tree.png")







#
