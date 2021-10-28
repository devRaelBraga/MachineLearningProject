import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl


def loadDataset(filename):
    print("Importando base de dados...")
    BaseDeDados = pd.read_csv(filename, delimiter= ";")
    X = BaseDeDados.iloc[:, :-1].values
    Y = BaseDeDados.iloc[:, -1].values
    return X, Y

def fillMissingdata(X):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values= np.nan, strategy= "median")
    X[:, 1:] = imputer.fit_transform(X[:, 1:])
    return X

def computeCategorization(X):
    from sklearn.preprocessing import LabelEncoder
    label_x = LabelEncoder()
    X[:, 0] = label_x.fit_transform(X[:, 0])

    #one-hot encoding
    D = pd.get_dummies(X[:, 0])
    X = X[:, 1:]
    X = np.insert(X, 0, D.values, axis=1)
    return X

def splitTrainTestSets(X, Y, testsize):
    from sklearn.model_selection import train_test_split
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = testsize )
    return XTrain, XTest, YTrain, YTest

def computeScaling(Train, Test):
    from sklearn.preprocessing import StandardScaler
    scaleX = StandardScaler()
    Train = scaleX.fit_transform(Train)
    Test = scaleX.fit_transform(Test)
    return Train, Test

def computeLinear(Xtrain,Xtest,Ytrain,Ytest):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(Xtrain, Ytrain)
    #Ypred = regressor.predict(Xtest)
    #print(Ytest, Ypred)
    plt.scatter(Xtest[:, -1], Ytest, color="red")
    plt.plot(Xtest[:, -1], regressor.predict(Xtest), color= "blue")
    plt.show()



def runLinearRegression(filename):
    X, Y = loadDataset(filename)
    X = fillMissingdata(X)
    X = computeCategorization(X)
    Xtrain, Xtest, Ytrain, Ytest = splitTrainTestSets(X, Y, 0.8)
    computeLinear(Xtrain, Xtest, Ytrain, Ytest)

if __name__ == "__main__":
    runLinearRegression("svbr.csv")