#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:20:48 2017

@author: zhenliu
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as op

#functions
def displaydata( X ):
    nrow = X.shape[0]
    ncol = X.shape[1]
    e_width = int(np.around( np.sqrt( ncol ) ))
#    print (e_width)
    e_height = int( ncol / e_width )
#    print (e_height)
    disp_rows = int(np.floor( np.sqrt( nrow ) ) )
    disp_cols = int(np.ceil( nrow / disp_rows ) )
    pad = 2
    disp_array = - np.ones(shape = (pad + disp_rows * (e_height + pad), pad + disp_cols * (e_width + pad)))
    curr = 0
    for j in range(disp_rows):
        for i in range(disp_cols):
            
            if curr > nrow :
                break;
            
#            mat = X[curr,:]
#            print (mat.shape)
            max_val = np.amax( np.absolute( X[curr,:] ) )
            temp_ex = np.reshape( (X[curr,:]),(e_height,e_width) ).T            
            temp_ex = temp_ex / max_val
            a = pad + j * (e_height + pad) 
            b = pad + j * (e_height + pad) + e_height
            c = pad + i * (e_width + pad)
            d = pad + i * (e_width + pad) + e_width
            disp_array[ a : b, c : d ] = temp_ex
            
            curr = curr + 1;
        if curr > nrow :
            break;
            

    plt.figure(figsize=(5.,5.))
    plt.axis('off')
    plt.imshow(disp_array, aspect='auto', cmap='gray', origin="upper")
    plt.savefig('fig_ex3.pdf', dpi=300, bbox_inches='tight')
    return();
    
def sigmoid( M ):
    sigm = 1/(1+np.exp(-M))
    return(sigm);
    
def lrCostFunction( theta, X, y ):
    nsamples = X.shape[0]
#    nfeatures = X.shape[1]
    h = sigmoid( np.dot( X,theta ) )
#    print ( h.shape, y.shape )
    cost = - y * np.log( h ) - ( 1 - y ) * ( np.log( 1- h ) )
#    print ( cost.shape )
    CF = np.sum( cost ) / nsamples
    GR = np.dot(X.T,( h - y )) / nsamples
#    print (GR.shape)

    return(CF,GR);
    
#cost-function and gradient using FOR loops from ex2_week3.py    
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
    
def lrCostFunctionReg( theta, X, y, lamda ):
    nsamples = X.shape[0]
    nfeatures = theta.shape[0]
    X = np.c_[np.ones(nsamples), X]
#    print(X.shape, theta.shape)
    h = sigmoid( np.dot( X, theta ) )
#    print ( h.shape, y.shape )
    J1 = ( - y * np.log( h ) - ( 1 - y ) * ( np.log( 1 - h ) ) ) / nsamples
    J2 = ( theta[1:nfeatures]**2 ) * ( lamda / ( 2 * nsamples ) ) 
#    print ( J1.shape, J2.shape )
    J = np.sum( J1 ) + np.sum( J2 )
    GR = X.T.dot( h - y ) / nsamples
#    print ( GR.shape )
    theta_r = theta * ( lamda / nsamples ) 
    theta_r[0] = 0
    GR = GR + theta_r
    return(J,GR);
    
def onevsall( X, y, nlabels, lamda ):
#    nsamples = X.shape[0]
#    nfeatures_old = X.shape[1]
#    X = np.c_[np.ones(nsamples), X]
#    print(X)
    nfeatures = X.shape[1] + 1
    theta0 = np.zeros( shape = (nfeatures,1) )
    theta_all = np.zeros( shape = (nlabels,nfeatures) )
    for i in range(nlabels):
        target = (y == i+1).astype(int)
#        print ('target', target.shape )
        theta_all[i,:] = op.minimize(
                fun = lrCostFunctionReg,
                x0 = theta0,
                args = (X, target, lamda),
                method = 'CG',
                jac = True,
                options = {
                'maxiter': 50,
                'disp': False,}).x
#    print(theta_all)
    return( theta_all );
    
def predictOVA( theta_all, X ):
    nsamples = X.shape[0]
#    nlabels = theta_all.shape[0]
    X = np.c_[ np.ones( nsamples ), X ]
    p = np.zeros( nsamples )
    A = sigmoid( np.dot( X, theta_all.T ) )
    p = np.argmax( A, axis=1 ) + 1
        
#    print ( p )
    return( p );
    
def predictNN( theta1, theta2, X ):
    nsamples = X.shape[0]    
    X = np.c_[ np.ones( nsamples ), X ]
    a1 = sigmoid( np.dot( theta1, X.T ) )
    a1_t = np.c_[ np.ones( nsamples ), a1.T ]
    a2 = sigmoid( np.dot( theta2, a1_t.T ) )
    p_nn = np.argmax( a2.T, axis=1 ) +1    
    return(p_nn);

#main program
D = sio.loadmat('ex3data1.mat')
W = sio.loadmat('ex3weights.mat')
#print ( D )
#print ( W['Theta1'] )
X = D['X']
y = D['y'].flatten()
theta1 = W['Theta1']
theta2 = W['Theta2']
#print ('X', X.shape, 'y', y.shape, 'theta1', theta1.shape, 'theta2', theta2.shape)


#plot a subset of data
#indices = np.random.permutation(nsamples)
#subX = X[indices[0:100],:]
#displaydata( subX )

#theta_init = np.zeros( shape=(401,1) )
#lrCostFunctionReg(theta_init, X, y, 1)

#train lr using one_vs_all & predict
#theta_all = onevsall( X, y, 10, 1 )
#p = predictOVA( theta_all, X )
#print ( np.mean( p == y ) )
#nsize = y.shape[0]
#print ( sum( (p == y).astype(int) )/nsize )

p_nn = predictNN( theta1, theta2, X )
print ( np.mean( p_nn == y ) )



#test cost function
#theta_t = np.array([-2,-1,1,2])
#X_t = (np.arange(15)+1).reshape(5,3)/10
#X_t = np.c_[np.ones(5), (np.arange(15)+1).reshape(5,3)/10]
#y_t = np.array([2,1,2,1,1])
#print ( lrCostFunction( theta0, a0, y0 ))
#print ( lrCostFunctionReg( theta_t, X_t, y_t, 3 ) )
#theta_all = onevsall( X_t, y_t, 2, 1 )
#p = predictOVA( theta_all, X_t )
