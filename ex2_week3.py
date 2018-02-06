#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:21:40 2017

@author: zhenliu
"""

import numpy as np
import pandas as pd
import scipy.optimize as op
import matplotlib.pyplot as plt

#def color(x):
#    if (x == 1.0):
#        return 'red'
#    else:
#        return 'blue'
#    return();
    
#def symbol(x):
#    if (x == 1.0):
#        return 'o'
#    else:
#        return '+'
#    return();
    
#xx = pd.Series([1.0, 0.0])
    
def sigmoid( M ):
    sigm = 1/(1+np.exp(-M))
    return(sigm);

#print (sigmoid(X))

def mapfeature( X1, X2 ):
    degree = 6
    n = degree + 1
    nfeatures = int((1 + degree + 1) * (degree+1)/2)
    nsamples = X1.shape[0]
#    print (nfeatures,nsamples)

    mf = np.ones(shape=(nsamples,nfeatures))
#    print (mf)
    kk=0
    for i in range(n):
        for j in range(i+1):
            A = X2**j
            B = X1**(i-j)
            mf[:,kk] = A*B
            kk = kk + 1;
            
#    print (mf)
    return(mf);

#print (mapfeature( X1, X2 ))
#print (mapfeature( A,B))

def costfunc( theta, X, y, lamda ):
    nfeatures = X.shape[1]
    nsamples = X.shape[0]
    Ja = 0.0
    Jb = 0.0
    for i in range(nsamples):
        hx = sigmoid( X[i,:].dot(theta) )
        Ja = Ja + (-(np.log(hx))*y[i]-(1-y[i])*(np.log(1-hx)))/nsamples;
    for j in range(1,nfeatures):
        Jb = Jb + lamda * ((theta[j])**2) / (2 * nsamples);
    J = Ja + Jb;
    return(J);

def gradient( theta, X, y, lamda ):
    nfeatures = X.shape[1]
    nsamples = X.shape[0]
    grad = np.zeros(shape=(nfeatures,1))
    for i in range(nsamples):
        hx = sigmoid( X[i,:].dot(theta) )
        grad[0] = grad[0] + (hx - y[i]) * X[i,0]/nsamples
    for j in range(1,nfeatures):
        for i in range(nsamples):
            hx = sigmoid( X[i,:].dot(theta) )
            grad[j] = grad[j] + (hx - y[i]) * X[i,j]/nsamples;
        grad[j] = grad[j] + lamda * theta[j]/nsamples;
    return(grad);

def plotdb( theta ):
    num = 50
    xa = np.linspace(-1.0, 1.25, num)
    xb = np.linspace(-1.0, 1.25, num)
    h = np.zeros(shape=(num,num))
    for i in range(num):
        ka = xa[i]
        a = np.array([ka])
        for j in range(num):
            kb = xb[j]
            b = np.array([kb])
            h[i,j] = np.dot(mapfeature( a, b ),theta);
    
    plt.contour(xa,xb,h.T,levels=[0.])       
#    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
#    plt.contourf(xa,xb,z)

    return();

fname = "ex2data2.txt"
R = np.loadtxt(fname,delimiter=',')
y = R[:,2]
X = R[:,:2]
X1 = X[:,0]
X2 = X[:,1]
K = mapfeature( X1, X2 )
lamda = 1
nfeatures = K.shape[1]
#print ('# of features is ', nfeatures);
theta = np.zeros(shape=(nfeatures,1))
#print ( costfunc( theta, K, y, lamda));
#print ( gradient( theta, K, y, lamda));
Result = op.minimize(fun = costfunc,x0 = theta, args = (K, y, lamda), method = 'TNC', jac = gradient);
optimal_theta = Result.x;
#print (optimal_theta);

#PLOT
plt.figure(figsize=(5.,5.))
plotdb( optimal_theta )
#plt.scatter(R[:,0], R[:,1], s=10, marker='o', c=pd.Series(R[:,2]).apply(lambda x: color(x)))
mapping1 = { 1 : '^', 0 : 'o' }
mapping2 = { 1 : 'red', 0 : 'blue'}
for i in range(len(y)):
    plt.scatter(X1[i], X2[i], s=20, marker=mapping1[y[i]], c=mapping2[y[i]] );
plt.xlabel('feature x1')
plt.ylabel('feature x2')
plt.savefig('fig_ex2.pdf',bbox_inches='tight')
plt.show()

#test
#A = np.array([1,2,3,4])
#B = np.array([2,3,4,5])
#print ((A-1)*B)
#test    
#xa = ([0,1,2])
#xb = ([1,2,3]) 
#k = xa[2]  
#a = np.array([k])