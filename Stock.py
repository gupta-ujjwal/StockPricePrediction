
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('AppleData.csv')

def modifyVal(dataset):
    X = [int(a[0].split('/')[2]) for a in dataset.iloc[5:, 0:1].values]
    y = dataset.iloc[5:, 1:2].values
    X = np.reshape(X,(len(X), 1))
    return X[::-1],y[::-1]

def trainModelLinear(X,y):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression() 
    regressor = regressor.fit(X ,y)
    y_pred = regressor.predict(X)
    print(regressor.predict(np.reshape([1,2,3,4,5,6],(6,1))))
    return y_pred

def trainModelSVR(X,y):
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X,y)
    y_pred = regressor.predict(X)
    print(regressor.predict(np.reshape([1,2,3,4,5,6],(6,1))))
    return y_pred

def plotter(X,y,y_predLinear, y_predSVR):
    plt.scatter(X, y, color="black", label="Points" )
    plt.plot(X, y_predLinear, color="Red", label="Linear Regressor" )
    plt.plot(X, y_predSVR, color="Pink", label="SVR Regressor" )
    plt.show()

X,y = modifyVal(dataset)
y_predLinear = trainModelLinear(X,y)
y_predSVR = trainModelSVR(X,y)
plotter(X,y,y_predLinear, y_predSVR)

#X = np.asarray(dates, dtype = float).transpose()[0:80]
#y = np.asarray(prices, dtype = float)[0:80]

#training the model
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X,y)
#
#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(np.asarray(dates, dtype = float).transpose()[80:], np.asarray(prices, dtype = float)[80:])
