import pandas as pd
import numpy as np
import ast
import itertools
# from sklearn.model_selection import train_test_split
import sklearn
from scipy import interp
import matplotlib.patches as mpatches
from sklearn import preprocessing, cross_validation, neighbors, linear_model
from sklearn.metrics import classification_report, precision_recall_curve
from math import sqrt
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import roc_curve, auc,confusion_matrix
# style.use('fivethirtyeight')
plt.style.use('ggplot')



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
    return [knn_model_1.score(X_test, y_test),knn_model_1.score(X_train, y_train)]


def optimalKNN(X_test,y_test,X_train,y_train,clf):

    print('==================KNN optimal=====================')
    print('k-NN score for test set: %f' % knn_model_2.score(X_test, y_test))
    print('k-NN score for training set: %f' % knn_model_2.score(X_train, y_train))
    y_true, y_pred = y_test, knn_model_2.predict(X_test)
    print(classification_report(y_true, y_pred))
    # y_scores = cross_val_predict(clf,X_train,y_train, cv=3)
    # precision, recall, threshold = precision_recall_curve(y_train, y_scores)
    return [knn_model_2.score(X_test, y_test),knn_model_2.score(X_train, y_train)]

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

    # df['sp'].replace('O', 0,inplace=True)
    # df['sp'].replace('B', 3,inplace=True)
    X = np.array(df.drop(['sp'],1))
    return [X,y]



if __name__ == '__main__':
    lis = preProcess()
    X, y= lis[0], lis[1]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.3)

    clf = neighbors.KNeighborsClassifier()

    knn_model_2 = clf.fit(X_train,y_train)
    values = optimalKNN(X_test,y_test,X_train,y_train,clf)


    knn_model_1 = diferentKValues(X_train,y_train,X_test,y_test,1)

    #
    y_score = knn_model_2.fit(X_train, y_train)


    #print decision confusion_matrix
    y_score_n = knn_model_2.fit(X_train, y_train).predict(X_test)
    conf_matrix = (confusion_matrix(y_test, y_score_n))
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=["B","O"],title='Confusion matrix, without normalization')


    #compute ROC
    probs = y_score.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds,pos_label='O')
    roc_auc = auc(fpr, tpr)

    predsB = probs[:,0]

    fprB, tprB, thresholdB = roc_curve(y_test, predsB,pos_label='B')
    roc_aucB = auc(fprB, tprB)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC for O = %0.2f' % roc_auc)
    plt.plot(fprB, tprB, 'r', label = 'AUC for B = %0.2f' % roc_aucB)

    plt.legend(loc = 'lower right')
    plt.plot([0, 1.1], [0, 1.1],'r--')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    # plt.ylabel('True 0 Rate')
    # plt.xlabel('False 0 Rate')
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.show()
    print(clf.classes_)
#determine best k
    # size_test_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # list_test, list_train = [],[]
    # i_value = []
    #
    #
    #
    # for i in range(1,40):
    #     knn_model_1 = diferentKValues(X_train,y_train,X_test,y_test,i)
    #     list_test.append(knn_model_1[0])
    #     list_train.append(knn_model_1[1])
    #     i_value.append(i)
    #
    #
    #
    # plt.plot(i_value,list_train,'r',i_value,list_test,'b')
    # plt.title('Variation of K value')
    # plt.ylabel('accuracy')
    # plt.xlabel('test size')
    # red_patch = mpatches.Patch(color='red', label='train set')
    # b_patch = mpatches.Patch(color='blue', label='test set')
    #
    # plt.legend(handles=[red_patch, b_patch])
    # plt.show()
    #
    #
    #







#determine best train size
    # size_test_list = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    # list_test, list_train = [],[]
    # i_value = []
    # for i in size_test_list:
    #     X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = i)
    #
    #     clf = neighbors.KNeighborsClassifier()
    #
    #     knn_model_2 = clf.fit(X_train,y_train)
    #
    #     values = optimalKNN(X_test,y_test,X_train,y_train,clf)
    #     list_test.append(values[0])
    #     list_train.append(values[1])
    #     i_value.append(i)
    #
    # plt.plot(i_value,list_train,'r',i_value,list_test,'b')
    # plt.title('Variation of train/test size')
    # plt.ylabel('accuracy')
    # plt.xlabel('test size')
    # red_patch = mpatches.Patch(color='red', label='train set')
    # b_patch = mpatches.Patch(color='blue', label='test set')
    #
    # plt.legend(handles=[red_patch, b_patch])
    # plt.show()
