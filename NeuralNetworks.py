import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

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

    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
    mlp.fit(X_train,y_train)

    print(mlp)

    predictions = mlp.predict(X_test)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

    # print(len(mlp.coefs_))
    # print(len(mlp.coefs_[0]))
    # print(len(mlp.intercepts_[0]))
