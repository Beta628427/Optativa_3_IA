# -*- coding: utf-8 -*-
"""
Spyder Editor
Classification problem for the AI course
author: @nikorose
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
#from sklearn.preprocessing import StandardScaler 
import warnings
warnings.filterwarnings("ignore")
## Function to plot

def plot_dataset(X,y,axes):
    plt.plot(X.iloc[:, 0][y==0], X.iloc[:, 1][y==0], "ro")
    plt.plot(X.iloc[:, 0][y==1], X.iloc[:, 1][y==1], "bo")
    if y.max() > 1:
        plt.plot(X.iloc[:, 0][y==2], X.iloc[:, 1][y==2], "go")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel('A', fontsize=20)
    plt.ylabel("B", fontsize=20, rotation=0)
    plt.show()
    return axes

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
#    try:
#        y_decision = clf.decision_function(X).reshape(x0.shape)
#        plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
#    except AttributeError:
#        pass

# =============================================================================
# Loading Iris data
# =============================================================================
iris = datasets.load_iris()
data = pd.DataFrame(iris.data)
targets = pd.DataFrame(iris.target)
targets4p = pd.DataFrame(np.array([0]*50 + [1]*100))
data_w_out = pd.DataFrame(pd.concat([data, targets, targets4p], axis=1, 
                                            ignore_index=True))
data_w_out.columns= ['len sepals','width sepals','len petal','width petal','output', 'reduced_output']

# define plotting function for future re-use
#ax1= data_w_out.iloc[:,:2].plot.scatter(x='len sepals',y='width sepals',c='output',
#                                     colormap='viridis')
#ax2= data_w_out.iloc[:,:2].plot.scatter(x='len sepals',y='width sepals',c='reduced_output',
#                                     colormap='viridis')

X= data_w_out.iloc[:,:4]
y= data_w_out.iloc[:,-2]
axes = np.array([np.min(X,axis=0)[0],np.max(X,axis=0)[0],np.min(X,axis=0)[1],np.max(X,axis=0)[1]])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.4)


classifiers = [
    ("svm_linear", SVC(kernel='linear')),
    ("svm_polynomial", SVC(kernel='poly')),
    ("svm_rbf", SVC(kernel='rbf')),
    ("logistic", LogisticRegression()),
    ("knn",KNeighborsClassifier()),
    ("decision_tree", DecisionTreeClassifier()),
    ('random_forest', RandomForestClassifier()),
    ('perceptron', MLPClassifier(alpha=1, max_iter=1000))]

param_grid = [
  {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']},
  {'C': [0.1, 1, 3], 'gamma': [0.1, 0.5], 'kernel': ['poly']},
  {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10], 'kernel': ['rbf']},
  {'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
  {'n_neighbors':range(1,50), 'leaf_size':range(5,61,5)},
  {'max_depth':range(1,50), 'min_samples_split':range(2,10)},
  {'max_depth':range(1,50), 'min_samples_split':range(2,10)},
  {'solver': ['lbfgs', 'sgd', 'adam'], 
   'activation' : ['identity', 'logistic', 'tanh', 'relu'],
   'hidden_layer_sizes': [(100,2),(100,3),(150,2),(150,3),(200,3)]}]

count = 0
for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    grid = GridSearchCV(clf, param_grid=param_grid[count], cv=5)
    grid.fit(X_train, y_train)
    print("Test set score: {:.2f}".format(grid.score(X_test, y_test))) 
    print("Best parameters: {}".format(grid.best_params_))
    if X.shape[1] <= 2:
        plot_predictions(grid, axes)
        plot_dataset(X,y,axes)
    count +=1

