#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:27:32 2019

@author: nikorose
"""

import numpy as np
from numpy import transpose as T
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron

iris = datasets.load_iris()
data = pd.DataFrame(iris.data)
targets = pd.DataFrame(iris.target)
targets4p = pd.DataFrame(np.array([-1]*50 + [1]*100))
data_reduced = data.iloc[:,0:2]
data_reduced_w_out = pd.DataFrame(pd.concat([data_reduced,targets, targets4p], axis=1, 
                                            ignore_index=True))
data_reduced_w_out.columns= ['len sepals','height sepals','output', 'reduced_output']

# define plotting function for future re-use
ax1= data_reduced_w_out.plot.scatter(x='len sepals',y='height sepals',c='output',
                                     colormap='viridis')
ax2= data_reduced_w_out.plot.scatter(x='len sepals',y='height sepals',c='reduced_output',
                                     colormap='viridis')

def plot(X,y,xx,yy,Z,n, c):
    # Put the result into a color plot
    fig, ax = plt.subplots()
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Accent)
    ax.axis('on')
    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=['red'], cmap=plt.cm.Paired)
    ax.set_title('Perceptron {} conv={}'.format(n,c))
    

n = 0
convergence = 1.0 
weights = np.zeros(2).astype('int')
Bias = []
while convergence != 0:
    n +=1
    Percep = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, 
                        max_iter=n, tol=None, shuffle=True, verbose=0,
                        eta0=1.0, random_state=42)
    Percep.fit(data_reduced_w_out.iloc[:,:2], data_reduced_w_out.iloc[:,-2])
    # create a mesh to plot in
    X = np.hstack((data_reduced_w_out['len sepals'].values.reshape(-1,1),
                   data_reduced_w_out['height sepals'].values.reshape(-1,1)))  
    y = data_reduced_w_out['reduced_output'].values.reshape(-1,1)
    h = .1 # plot step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = Percep.predict(np.c_[xx.ravel(), yy.ravel()])
    weights = np.vstack((weights,Percep.coef_[0].astype('int')))
    Bias.append(Percep.intercept_[0].astype('int'))
    convergence = sum(weights[-1] - weights[-2])
    plot(X,y,xx,yy,Z,n, convergence)
compilated = np.hstack((weights[1:,:],np.array(Bias).reshape(-1,1)))
Export_data = pd.DataFrame(compilated)
