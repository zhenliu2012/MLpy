#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:11:15 2017

@author: zhenliu
"""
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import libsvm
from sklearn import svm
import numpy as np

def plotdata( X, y ):
    X1 = X[:,0]
    X2 = X[:,1]
    mapping1 = { 1 : '^', 0 : 'o' }
    mapping2 = { 1 : 'red', 0 : 'blue'}
    for i in range(len(y)):
        plt.scatter( X1[i], X2[i], s=20, marker=mapping1[y[i]], c=mapping2[y[i]] );
    return();
 
def SVMlinear1( X, y, C_p ):
#for practice
    model = libsvm.fit(X, y, kernel='linear', tol=0.001, C=C_p, max_iter=25);
    support_vectors = model[1]
    coeffs = model[3]
    intercept = model[4]
    normal_vector = coeffs.dot(support_vectors).flatten()
    xp = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 100)
    yp = -(normal_vector[0] * xp + intercept) / normal_vector[1]
    return(xp, yp);
    
def SVMlinear2( X, y, C ):
    clf = svm.SVC(kernel='linear', C=C);
    clf.fit(X, y);
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-0.4, 4.5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, 'k-')
#plot decision boundary    
    h = 0.02
    k = 0.5
    x1_min, x1_max = X[:,0].min() - k, X[:,0].max() + k
    x2_min, x2_max = X[:,1].min() - k, X[:,1].max() + k
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = clf.predict( np.c_[xx1.ravel(), xx2.ravel()] ).reshape( xx1.shape )
    plt.pcolormesh( xx1, xx2, Z, cmap = plt.cm.Paired )
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
            facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')
    plt.show()
    plt.axis('tight')
    return();
    
def SVMrbf( X, y, C, sigma ):
#    sigma = 0.1
    gamma = 1.0 / (2.0 * sigma ** 2)
    clf = svm.SVC(kernel = 'rbf', C=C, gamma = gamma)
    clf.fit(X, y)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
    h = 0.01
    k = 0.05
    x1_min, x1_max = X[:,0].min() - k, X[:,0].max() + k
    x2_min, x2_max = X[:,1].min() - k, X[:,1].max() + k
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = clf.predict( np.c_[xx1.ravel(), xx2.ravel()] ).reshape( xx1.shape )
    plt.pcolormesh( xx1, xx2, Z, cmap = plt.cm.Paired )
# Plot also the training points and support vectors
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set3, edgecolors='k')
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
        facecolors='none', zorder=10, edgecolors='k')
    plt.title('2-Class classification using Support Vector Machine with RBF kernel')
    plt.axis('tight')
    plt.show()
    return();

def gaussianKernel( X1, X2, sigma ):
    g = np.exp ( - np.sum( ( X1 - X2 )**2 ) / ( 2 * (sigma**2) ) )
    return(g);
def gaussianGramMatrix( X1, X2 ):
    sigma = 0.1
#    print ( X1.shape[0] )
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
#            print ( i, x1, j, x2 )
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(- np.sum( np.power((x1 - x2),2) ) / float( 2*(sigma**2) ) )
    return(gram_matrix);
    
def SVMgaussian( X, y, C ):
#custom gaussian kernel    
    clf = svm.SVC( kernel = 'precomputed', C = C )
    clf.fit(gaussianGramMatrix( X, X ), y)
    
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
    h = 0.01
    k = 0.05
    x1_min, x1_max = X[:,0].min() - k, X[:,0].max() + k
    x2_min, x2_max = X[:,1].min() - k, X[:,1].max() + k
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Xpred = np.c_[xx1.ravel(), xx2.ravel()]
#    print (Xpred.shape)
    Z = clf.predict( gaussianGramMatrix( Xpred, X ) ).reshape( xx1.shape )
    plt.pcolormesh( xx1, xx2, Z, cmap = plt.cm.Paired )
# Plot also the training points and support vectors
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set3, edgecolors='k')

#    print ( clf.support_vectors_ ) #returns empty array [ ] with custom kernel
    plt.title('2-Class classification using Support Vector Machine with custom kernel')
    plt.axis('tight')
    plt.show()    
    return();
def trainSVM( X, y, Xval, yval ):
    C = np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ]).flatten()
    sigma = np.array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ]).flatten()
    gamma = 1.0 / (2.0 * sigma ** 2)
    results = []
    for i in range( 8 ):
        for j in range( 8 ):
            Cc = C[i]
            Gg = gamma[j]
            Sig = sigma[j]
            clf = svm.SVC( kernel = 'rbf', C=Cc, gamma = Gg )
            clf.fit(X, y)
            Z = clf.predict( Xval )
            err = np.mean( Z == yval )
#            print ( np.mean( Z == yval ), Cc, Sig )
            results.append( [ err, Cc, Sig ] )
    output = np.vstack(results) 
    index = np.argmax( output[:,0] )
    C_opt, Sig_opt = output[index, 1], output[index, 2]
    return(C_opt, Sig_opt);
    
        
#main program
D1 = sio.loadmat('ex6data1.mat')
D2 = sio.loadmat('ex6data2.mat')
D3 = sio.loadmat('ex6data3.mat')
#X = D['X']
X1 = np.require(D1['X'], dtype = np.float64, requirements=['C'])
y1 = np.require(D1['y'].flatten(), dtype = np.float64)
X2 = np.require(D2['X'], dtype = np.float64, requirements=['C'])
y2 = np.require(D2['y'].flatten(), dtype = np.float64)
X3 = np.require(D3['X'], dtype = np.float64, requirements=['C'])
y3 = np.require(D3['y'].flatten(), dtype = np.float64)
Xval = np.require(D3['Xval'], dtype = np.float64, requirements=['C'])
yval = np.require(D3['yval'].flatten(), dtype = np.float64)
#plotdata( X2, y2 );
SVMlinear2( X1, y1, 1 );
#SVMrbf( X2, y2, 1.0, 0.1 );
#SVMgaussian( X2, y2, 1.0 );
#plotdata( X3, y3 );
C_opt, Sigma_opt = trainSVM( X3, y3, Xval, yval )
SVMrbf( X3, y3, C_opt, Sigma_opt );

#test
#x1 = np.array([1, 2, 1])
#x2 = np.array([0, 4, -1]); 
#sigma = 2;
#print ( gaussianKernel(x1, x2, sigma) );
#gram_matrix = gaussianGramMatrix( X2, X2 )