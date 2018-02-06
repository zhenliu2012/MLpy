#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 22:14:30 2017

@author: zhenliu
"""

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

def coll(x):
    if (x == 1.0):
        return 'red'
    else:
        return 'blue'
    return();
    
#xx = pd.Series([1.0, 0.0])
def normalize( X ):
    n = X.shape[1]
#    m = X.shape[0]
#    print ('dim', n,m)
    for i in range(n):
        Xc = X[:,i]
        Xc = (Xc - Xc.mean())/np.std(Xc)
        X[:,i] = Xc;
        
#    X = np.c_[np.ones(m),X]
#    print ( X )
    return(X);
    
def sigmoid( M ):
    sigm = 1/(1+np.exp(-M))
    return(sigm);

#print (sigmoid(X))

def mapfeature( X1, X2 ):
    degree = 6
    n = degree + 1
    nfeatures = int((1 + degree + 1) * (degree+1)/2)
    nsamples = len(X1)
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

def costfunc( theta, X, y ):
#    nfeatures = X.shape[1]
    nsamples = X.shape[0]
    Ja = 0.0
#    Jb = 0.0
    for i in range(nsamples):
        hx = sigmoid( X[i,:].dot(theta) )
        Ja = Ja + (-(np.log(hx))*y[i]-(1-y[i])*(np.log(1-hx)))/nsamples;
#    for j in range(1,nfeatures):
#        Jb = Jb + lamda * ((theta[j])**2) / nsamples;
#    J = Ja + Jb;
    return(Ja);

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


fname = "ex2data2.txt"
R = np.loadtxt(fname,delimiter=',')
y = R[:,2]
X = R[:,:2]
X = normalize( X )
X1 = X[:,0]
X2 = X[:,1]
plt.scatter(R[:,0], R[:,1], s=50, marker='o', c=pd.Series(R[:,2]).apply(lambda x: coll(x)))
plt.xlabel('feature x1')
plt.ylabel('feature x2')
plt.show()
#test
#A = np.array([1,2,3,4])
#B = np.array([2,3,4,5])
#print (A*B)
    
K = mapfeature( X1, X2 )
lamda = 1
nfeatures = K.shape[1]
#print ('# of features is ', nfeatures);
theta = np.zeros(shape=(nfeatures,1))
#print ( costfunc( theta, K, y, lamda));
#print ( gradient( theta, K, y, lamda));
Result = op.minimize(fun = costfunc,x0 = theta, args = (K, y, lamda), method = 'TNC', jac = gradient);
optimal_theta = Result.x;
print (optimal_theta);
        