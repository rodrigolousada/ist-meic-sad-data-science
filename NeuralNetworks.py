import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
import itertools
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

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.

    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

if __name__ == '__main__':
    balanced_data = pd.read_csv('crabs.csv',sep= ',', header= None)

    lis = preProcess()
    X, Y= lis[0],lis[1]

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

    ###################
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    print(scaler)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    ###################

    # for i in range(1,5):
    #     for j in range(1,15):
    #         param=()
    #         for i1 in xrange(i):
    #             param+=(i)
    mlp = MLPClassifier(hidden_layer_sizes=(9,9,9), max_iter=500)
    y_score=mlp.fit(X_train,y_train)

    print(mlp)

    predictions = mlp.predict(X_test)

    conf_matrix = confusion_matrix(y_test,predictions)
    print(conf_matrix)
    print(classification_report(y_test,predictions))


    #print decision confusion_matrix
    # plt.figure()
    # plot_confusion_matrix(conf_matrix, classes=["B","O"],title='Confusion matrix, without normalization')
    # print(len(mlp.coefs_))
    # print(len(mlp.coefs_[0]))
    # print(len(mlp.intercepts_[0]))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9, [7, 9, 9, 9, 2])
    fig.savefig('nn.png')

    # #compute ROC
    # probs = y_score.predict_proba(X_test)
    # preds = probs[:,1]
    # fpr, tpr, threshold = roc_curve(y_test, preds,pos_label='O')
    # roc_auc = auc(fpr, tpr)
    #
    # predsB = probs[:,0]
    #
    # fprB, tprB, thresholdB = roc_curve(y_test, predsB, pos_label='B')
    # roc_aucB = auc(fprB, tprB)
    #
    # # method I: plt
    # import matplotlib.pyplot as plt
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label = 'AUC for O = %0.2f' % roc_auc)
    # plt.plot(fprB, tprB, 'r', label = 'AUC for B = %0.2f' % roc_aucB)
    #
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1.1], [0, 1.1],'r--')
    # plt.xlim([0, 1.1])
    # plt.ylim([0, 1.1])
    # # plt.ylabel('True 0 Rate')
    # # plt.xlabel('False 0 Rate')
    # plt.xlabel("FPR", fontsize=14)
    # plt.ylabel("TPR", fontsize=14)
    # plt.show()
    # print(clf.classes_)
