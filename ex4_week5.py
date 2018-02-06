#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:43:21 2017

@author: zhenliu
"""

import numpy as np
import scipy.io as sio
import scipy

def sigmoid( M ):
    sigm = 1/(1+np.exp(-M))
    return(sigm);
    
def sigmoidgrad( M ):
    grad = 1 / ( 2 + np.exp( -M )+ np.exp( M ) )
    return(grad);
    
def costfunction( nn_params, input_size, hidden_size, nlabels, X, y, lambdaa ):
    nsamples = X.shape[0]
    J1 = 0. 
#    D1 = np.zeros( shape = (hidden_size,input_size+1) )
#    D2 = np.zeros( shape = (nlabels,hidden_size+1) )
    X = np.c_[ np.ones( nsamples ), X ]
    theta1 = nn_params[0:hidden_size*(input_size+1)].reshape(hidden_size,input_size+1)
    theta2 = nn_params[hidden_size*(input_size+1):hidden_size*(
            input_size+1)+nlabels*(hidden_size+1)].reshape(nlabels,hidden_size+1)
    
    for i in range(nsamples):
        a1 = X[i,:]
        a2 = sigmoid( np.dot( theta1, a1 ) )
        a2_b = np.concatenate( ( ([1.]), a2 ), axis = 0 )
        a3 = sigmoid( np.dot( theta2, a2_b ) )
        yy = np.zeros(nlabels)
        yy[y[i]-1] = 1
#        delta3 = a3 - yy
#        delta2 = np.dot(theta2.T, delta3) * ( a2_b * ( 1 - a2_b ) )
#        D1 = D1 + np.dot( delta2[1:].reshape(hidden_size,1), a1[np.newaxis] ) / nsamples
#        D2 = D2 + np.dot( delta3.reshape(nlabels,1), a2_b[np.newaxis]) / nsamples      
        J1 = J1 + np.sum (
                ( - yy * np.log( a3 ) - ( 1 - yy ) * ( np.log( 1 - a3 ) ) ) / nsamples )
    
    J = J1 + ( np.sum( theta1[:,1:input_size+1]**2 ) + np.sum( 
            theta2[:,1:hidden_size+1]**2) ) * ( lambdaa / (2 * nsamples ) )
#    Greg1 = ( lambdaa / nsamples ) * theta1
#    Greg1[:,0] = np.zeros( hidden_size )
#    Greg2 = ( lambdaa / nsamples ) * theta2
#    Greg2[:,0] = np.zeros( nlabels )
#    D1 = D1 + Greg1
#    D2 = D2 + Greg2
#    print ( D1.shape, D2.shape )
#    grad = np.concatenate( (D1.flatten(), D2.flatten()), axis=0 )
    
    return(J); 
    
def nncostfunction( nn_params, input_size, hidden_size, nlabels, X, y, lambdaa ):
    nsamples = X.shape[0]
    J1 = 0. 
    D1 = np.zeros( shape = (hidden_size,input_size+1) )
    D2 = np.zeros( shape = (nlabels,hidden_size+1) )
    X = np.c_[ np.ones( nsamples ), X ]
    theta1 = nn_params[0:hidden_size*(input_size+1)].reshape(hidden_size,input_size+1)
    theta2 = nn_params[hidden_size*(input_size+1):hidden_size*(
            input_size+1)+nlabels*(hidden_size+1)].reshape(nlabels,hidden_size+1)
    
    for i in range(nsamples):
        a1 = X[i,:]
        a2 = sigmoid( np.dot( theta1, a1 ) )
        a2_b = np.concatenate( ( ([1.]), a2 ), axis = 0 )
        a3 = sigmoid( np.dot( theta2, a2_b ) )
        yy = np.zeros(nlabels)
        yy[y[i]-1] = 1
        delta3 = a3 - yy
        delta2 = np.dot(theta2.T, delta3) * ( a2_b * ( 1 - a2_b ) )
        D1 = D1 + np.dot( delta2[1:].reshape(hidden_size,1), a1[np.newaxis] ) / nsamples
        D2 = D2 + np.dot( delta3[:,None], a2_b[None,:]) / nsamples      
        J1 = J1 + np.sum (
                ( - yy * np.log( a3 ) - ( 1 - yy ) * ( np.log( 1 - a3 ) ) ) / nsamples )
    
    J = J1 + ( np.sum( theta1[:,1:input_size+1]**2 ) + np.sum( 
            theta2[:,1:hidden_size+1]**2) ) * ( lambdaa / (2 * nsamples ) )
    Greg1 = ( lambdaa / nsamples ) * theta1
    Greg1[:,0] = np.zeros( hidden_size )
    Greg2 = ( lambdaa / nsamples ) * theta2
    Greg2[:,0] = np.zeros( nlabels )
    D1 = D1 + Greg1
    D2 = D2 + Greg2
#    print ( D1.shape, D2.shape )
    grad = np.concatenate( (D1.flatten(), D2.flatten()), axis=0 )
    
    return(J,grad);
    
def numericalgrad( theta, X, y, lambdaa ):
    print ( 'computing numerical gradient' )
    numgrad = np.zeros( len(theta) )
    per = np.zeros( len(theta) )
    e = 1e-4
    for i in range( len(theta) ):
        print ( 'current iteration #', i +1 )
        per[i] = e
        Ja = costfunction( theta + per, 400, 25, 10, X, y, lambdaa )
        Jb = costfunction( theta - per, 400, 25, 10, X, y, lambdaa )
        numgrad[i] = (Ja - Jb ) / (2 * e)
        per[i] = 0. 
        
    return(numgrad);

def nncostfunctionVEC( nn_params, input_size, hidden_size, nlabels, X, y, lambdaa ):
    nsamples = X.shape[0]
    J1 = 0. 
    X = np.c_[ np.ones( nsamples ), X ]
    theta1 = nn_params[0:hidden_size*(input_size+1)].reshape(hidden_size,input_size+1)
    theta2 = nn_params[hidden_size*(input_size+1):hidden_size*(
            input_size+1)+nlabels*(hidden_size+1)].reshape(nlabels,hidden_size+1)
    a2 = sigmoid( np.dot( theta1, X.T ) )
    a2_b = np.concatenate( ( ([1.]), a2 ), axis = 0 )
    a3 = sigmoid( np.dot( theta2, a2_b ) )
    
    return();

def predictNN( theta1, theta2, X ):
    nsamples = X.shape[0]    
    X = np.c_[ np.ones( nsamples ), X ]
    a1 = sigmoid( np.dot( theta1, X.T ) )
    a1_t = np.c_[ np.ones( nsamples ), a1.T ]
    a2 = sigmoid( np.dot( theta2, a1_t.T ) )
    p_nn = np.argmax( a2.T, axis=1 ) +1    
    return(p_nn);

def randInitializeWeights(L_in, L_out):
    epsilon = 0.12
    W = np.random.rand( L_out, L_in + 1 ) * 2 * epsilon - epsilon
    return( W );
    
#main program
D = sio.loadmat('ex4data1.mat')
W = sio.loadmat('ex4weights.mat')
#print ( D )
#print ( W['Theta1'] )
X = D['X']
y = D['y'].flatten()
theta1 = W['Theta1']
theta2 = W['Theta2']
#print ('X', X.shape, 'y', y.shape, 'theta1', theta1.shape, 'theta2', theta2.shape)
nn_params = np.concatenate((theta1.flatten(),theta2.flatten()),axis=0)
theta1_initial = randInitializeWeights(400, 25)
theta2_initial = randInitializeWeights(25, 10)
nn_params_initial = np.concatenate(
        (theta1_initial.flatten(),theta2_initial.flatten()),axis=0)
#print ( theta1_initial.shape, theta2_initial.shape )
theta = scipy.optimize.minimize(
        fun = nncostfunction,
        x0 = nn_params_initial,
        args = (400, 25, 10, X, y, 1),
        method = 'CG',
        jac = True,
        options = {
        'maxiter': 200,
        'disp': False,}).x
hidden_size = 25
input_size = 400
nlabels = 10
theta1_nn = theta[0:hidden_size*(input_size+1)].reshape(hidden_size,input_size+1)
theta2_nn = theta[hidden_size*(input_size+1):hidden_size*(
            input_size+1)+nlabels*(hidden_size+1)].reshape(nlabels,hidden_size+1)
p_nn = predictNN( theta1_nn, theta2_nn, X )
print ( np.mean( p_nn == y ) )

#print ( nncostfunction( nn_params, 400, 25, 10, X, y, 1 ))
#numgrad = numericalgrad( nn_params, X, y, 3 )
#[J, grad] = nncostfunction( nn_params, 400, 25, 10, X, y, 3 )
#print ( np.linalg.norm( numgrad - grad ) / np.linalg.norm( numgrad + grad ) )
#print ( sigmoidgrad( np.ones(5)))
#print ( randInitializeWeights( 20, 10 ))