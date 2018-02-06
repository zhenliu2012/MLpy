#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:00:15 2017

@author: zhenliu
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
#import pandas as pd
#file = open("ex1data2.txt", "r") 
#for line in file: 
#    print (line);

#raw data including x1,x2,y
fname = "ex1data2.txt"
R = np.loadtxt(fname,delimiter=',')
#print (R)
#extract y
y = R[:,2]
m=len(y)
#print (y)
#extract X
X = R[:,:2]
#X = np.c_[X,np.random.rand(m,1)]
#print (X.shape[1])
#print (X)
theta = ([0,0,0])
#extract x1, x2
#Xa = X[:,0]
#Xb = X[:,1]
#print (Xa)
#print (Xb)
#demean
#meana = Xa.mean()
#meanb = Xb.mean()
#print (meana,meanb)
#Xa = Xa - Xa.mean()
#Xb = Xb - Xb.mean()
#print ( Xa )
#stda = np.std(Xa)
#print ('stda',stda)
#Xa = Xa/np.std(Xa)
#Xb = Xb/np.std(Xb)
#combine 
#Xn = np.vstack((Xa,Xb)).T
#print (Xn)
#Xn = np.c_[np.ones(m),Xn]
#print (Xn)
#print ("AAAAAAAAAAAAA")
def normalize( X ):
    n = X.shape[1]
    m = X.shape[0]
#    print ('dim', n,m)
    for i in range(n):
        Xc = X[:,i]
        Xc = (Xc - Xc.mean())/np.std(Xc)
        X[:,i] = Xc;
        
    X = np.c_[np.ones(m),X]
#    print ( X )
    return(X);
    
def graddesc( X, y, theta, alpha, niter ):
    
    n = X.shape[1]
    m = X.shape[0]   
#    print ('dim-norm:', '#features=',n,'#samples=',m)

    J = np.zeros(niter)
#    J = np.zeros(shape=(niter,n))
#    print (J);
    for k in range(niter):
#        print (theta)
        for j in range(n):
            sm = 0.0
            for i in range(m):
#                print(X[i,j])
                sm = sm+(X[i,:].dot(theta)-y[i])*X[i,j];
        
            theta[j] = theta[j]-(alpha/m)*sm;
            
        J[k] = costf(X,y,theta)
#        J[k,1] = k+1
#        print (J[k])

#    print (theta)
    return(theta,J);
      
def costf( X, y, theta ):
   "This computes the cost function"
   m=len(y)
#   print (m)
   J=0
   for i in range(m):
#       print (X[i,:].dot(theta))
       J=J+(X[i,:].dot(theta)-y[i])**2;
       
   J=J/(2*m)  
   return(J);

def normeqn( X, y ):
    XT = X.transpose()
    A = inv(np.dot(XT,X))
    B = np.dot(A,XT)
    theta = np.dot(B,y)
    
    return(theta);
#
X2 = normalize(X)
theta2,Jcost = graddesc( X2,y,theta,0.02,1000)
print (theta2)
#print (np.arange(1000)+1)
plt.plot(np.arange(1000)+1,Jcost)
plt.ylabel('cost function')
plt.xlabel('No. of iterations')
plt.show()

X3 = np.c_[np.ones(m),X]
print (normeqn(X3,y))
    
    
    
    
    
    
    
    
    
    
    
    