#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:20:40 2018

@author: zhenliu
"""

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def loaddata( name, plotflag ):
    D = spio.loadmat( name )
    var = spio.whosmat( name )
    X1 = np.require(D[var[0][0]], dtype = np.float64, requirements=['C'])
    X2 = np.require(D[var[1][0]], dtype = np.float64, requirements=['C'])
    X3 = np.require(D[var[2][0]], dtype = np.float64, requirements=['C'])
    if plotflag == 1 :
        plt.scatter(X1[:, 0], X1[:, 1], s = 30, marker = '^', edgecolors='black', zorder=20 )
        plt.axis( 'scaled' )
        plt.show()
    return( X1, X2, X3 );

def estimateGaussian( X ):
    mu = np.mean( X, axis = 0 )
    sigma2 = np.var( X, axis = 0 )
    cov = np.diag( sigma2 )
    return( mu, cov );
    
def multivariateGaussian( X, mu, cov, plotflag ):
    rv = multivariate_normal( mu, cov )
    if plotflag == 1:
        k = 5
        h = 0.5 #step
        x1_min, x1_max = X[:,0].min() - k, X[:,0].max() + k
        x2_min, x2_max = X[:,1].min() - k, X[:,1].max() + k
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
        pos = np.empty(xx1.shape + (2,))
        pos[:, :, 0] = xx1
        pos[:, :, 1] = xx2
        p = rv.pdf( pos )    
        plt.scatter(X[:, 0], X[:, 1], s = 20, marker = 'x', color ='black', linewidth = 0.75, zorder=0 )
        plt.contour(xx1, xx2, p, levels = np.power(10,np.arange(-20.,0.,3.)), colors = 'b' )
        plt.axis( 'scaled' )
        plt.show()

    return(rv.pdf( X ));
    
def selectthreshold( yval, pval ):
    stepsize = ( pval.max() - pval.min() ) / 1000
#    print ( pval.min(), pval.max(), stepsize )
    F1_best = 0.
    for e in np.arange( pval.min() + stepsize, pval.max(), stepsize ):
        predictions = ( pval < e ).astype(int)
        fp = np.sum( (( predictions == 1 )&( yval == 0 )).astype(int) )
        tp = np.sum( (( predictions == 1 )&( yval == 1 )).astype(int) )
        fn = np.sum( (( predictions == 0 )&( yval == 1 )).astype(int) )
        precision = tp / ( tp + fp )
        recall = tp / ( tp + fn )
        F1 = 2 * precision * recall / ( precision + recall )
#        print( e, F1 )
        if F1 > F1_best:
            F1_best = F1
            e_best = e
    return( e_best, F1_best );    
def anomalydetection( filename, plotflag ):
    print( 'data read in: ', filename )
    X, Xval, yval = loaddata( filename, 0 )
    yval = yval.flatten()
#estimate mean and var using the training set
    mu, cov = estimateGaussian( X )
    p = multivariateGaussian( X, mu, cov, 0  )
#cross-validation set --> find threshold
    pval = multivariateGaussian( Xval, mu, cov, 0 )
    e_best, F1_best = selectthreshold( yval, pval )
    print( 'threshold epsilon = ', e_best )
    print( 'F1 score = ', F1_best )
    print ( '# of outliers', np.sum( ( p < e_best ).astype(int) ) )
    if plotflag == 1:
        outliers = np.where( p < e_best )
        plt.scatter(X[:, 0], X[:, 1], s =20, marker = 'x', color='black', linewidth = 0.75 )
        plt.scatter(X[outliers, 0], X[outliers, 1], s=100, marker = 'o', facecolors='none', edgecolor = 'b', linewidth = 1.75  )
        plt.show()
    return();        

if __name__ == "__main__":
    anomalydetection( 'ex8data1.mat', 1 )
    anomalydetection( 'ex8data2.mat', 0 )

    