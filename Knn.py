import pandas as pd
import numpy as np
import ast
# from sklearn.model_selection import train_test_split
import sklearn
from sklearn import preprocessing, cross_validation, neighbors, linear_model
from sklearn.metrics import classification_report, precision_recall_curve
from math import sqrt
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
# style.use('fivethirtyeight')
plt.style.use('ggplot')





def presentHist(df):
    pd.DataFrame.hist(df, figsize = [12,12])
    df.plot.hist()
    plt.show()

def diferentKValues(X_train,y_train,X_test,y_test,kvalue):
    print('==================KNN k=' +str(kvalue) +'=====================')

    knn = neighbors.KNeighborsClassifier(n_neighbors = kvalue)
    knn_model_1 = knn.fit(X_train, y_train)

    print('k-NN score for test set: %f' % knn_model_1.score(X_test, y_test))
    print('k-NN score for training set: %f' % knn_model_1.score(X_train, y_train))


    y_true, y_pred = y_test, knn_model_1.predict(X_test)
    print(classification_report(y_true, y_pred))
    return knn_model_1


def optimalKNN(X_test,y_test,X_train,y_train,clf):

    print('==================KNN optimal=====================')
    print('k-NN score for test set: %f' % knn_model_2.score(X_test, y_test))
    print('k-NN score for training set: %f' % knn_model_2.score(X_train, y_train))
    y_true, y_pred = y_test, knn_model_2.predict(X_test)
    print(classification_report(y_true, y_pred))
    # y_scores = cross_val_predict(clf,X_train,y_train, cv=3)
    # precision, recall, threshold = precision_recall_curve(y_train, y_scores)

def preProcess():
    df = pd.read_csv('crabs.csv')
    #if data is not complete
    # df.replace('?', -99999, inplace=True)

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
    lis = preProcess()
    X, y= lis[0], lis[1]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

    clf = neighbors.KNeighborsClassifier()

    knn_model_2 = clf.fit(X_train,y_train)

    optimalKNN(X_test,y_test,X_train,y_train,clf)


    knn_model_1 = diferentKValues(X_train,y_train,X_test,y_test,100)












#
