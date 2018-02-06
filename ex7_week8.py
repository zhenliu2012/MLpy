#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:45:42 2018

@author: zhenliu
"""
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
from PIL import Image
from sklearn.preprocessing import scale

def centassgn( X, c_init ):
    n = X.shape[0]
    K = c_init.shape[0]
    c_idx = np.zeros(( n ))
    for i in range( n ):
        kk = np.zeros(( K ))
        for j in range( K ):
            kk[j] = np.sum( ( X[i,:] - c_init[j,:] )**2 );
        c_idx[i] = np.argmin( kk ).astype(int)
        
    return( c_idx );

def computecent( X, cent, idx ):
#    n = X.shape[0]
    m = X.shape[1]
    K = cent.shape[0]
#    print ( idx.shape )
    means = np.zeros((cent.shape[0], cent.shape[1]))
    for i in range( K ):
        c_i = (idx==i).astype(int)
#        print( c_i.shape)
        n_i = np.sum( c_i )
#        print( n_i )
        if n_i == 0 :
            print ( 'no data points assigned to this centroid' )
            break;
        c_i_mat = np.matlib.repmat( c_i, m, 1 ).T
        X_c_i = X * c_i_mat
        means[i,:] = np.sum( X_c_i, axis = 0 ) / n_i 
#    for i in range( K ):
#        count = 0
#        for j in range( n ):
#            if idx[ j ] == i:
#                count = count + 1
#                means[i,:] = means[i,:] + X[j,:]        
#        means[i,:] = means[i,:] / count
    return( means );
                
def runkmeans( X, c_init, maxiters, plotflag ):
    c = c_init
    if plotflag == 1:        
        plt.scatter(c[:, 0], c[:, 1], s = 50, facecolors='none', edgecolors='black', zorder=20 )
        colors = iter(cm.Reds(np.linspace(0, 1, maxiters)))
    for i in range(maxiters):
#        print( 'current iteration #', i + 1, '/', maxiters )
        idx = centassgn( X, c )
        c = computecent( X, c, idx )
        if plotflag == 1:
            plt.scatter(c[:, 0], c[:, 1], s = 50, c=next(colors), zorder=10 )
    if plotflag == 1:
        plt.scatter( X[:,0], X[:,1], c = idx, alpha=0.1, s=30, zorder=0 )
    return(c, idx);

def randinit( X, K ):
    indices = np.random.permutation(X.shape[0])
    c_init = X[indices[0:K],:]
    return(c_init);

def imagecompression( image, K ):
    if image.lower().endswith( '.png' ):
        A = imageio.imread( image )
#        print(A.shape)
#slice to drop the alpha channel if RGBA
        A = A[:,:,:3]
#normalize RGB values to 0-1
        A = A / 255
        X = A.reshape( A.shape[0] * A.shape[1], A.shape[2] )
        plt.subplot( 1, 2, 1 )
        plt.imshow(A, aspect='auto', origin="upper")
        plt.title( 'orginal png image' )
        plt.xticks([], [])
        plt.yticks([], [])
        plt.axis('scaled')
    elif image.lower().endswith( '.mat' ):
        Dpic = spio.loadmat( image )
        var = spio.whosmat( image )
        A = np.require(Dpic[var[0][0]], dtype = np.float64, requirements=['C'])
        A = A / 255
        plt.subplot( 1, 2, 1 )
        plt.imshow(A, aspect='auto', origin="upper")
        plt.title( 'orginal png image' )
        plt.xticks([], [])
        plt.yticks([], [])
        plt.axis('scaled')
        X = A.reshape( A.shape[0] * A.shape[1], A.shape[2] )
#    print( 'image compression using K-means algorithm' )
    c_init = randinit( X, K )
    c, idx = runkmeans( X, c_init, 10, 0 )
    X_rec = np.zeros( (X.shape ) )
    n = X.shape[0]
    for i in range( n ):
        k = idx[ i ].astype(int)
        X_rec[i,:] = c[k,:]
    X_rec = X_rec.reshape( (A.shape[0], A.shape[1], A.shape[2]) )
#make the plot
    plt.subplot( 1, 2, 2 )    
    plt.imshow(X_rec, aspect='auto', origin="upper")
    plt.title( 'compressed image [ K = %i ]' %K )
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('scaled')
    plt.tight_layout()
    plt.savefig('fig_ex7_compressed.png', dpi=300, bbox_inches='tight')
    plt.show()
    return();

def loaddata( name ):
    D = spio.loadmat( name )
    var = spio.whosmat( name )
    X = np.require(D[var[0][0]], dtype = np.float64, requirements=['C'])
    return( X );
    
def kmeansclustering( name, K, maxiters, plotflag ):
    X = loaddata( name )
    c_init = randinit( X, K )
    c, idx = runkmeans( X, c_init, maxiters, plotflag )
    return( c, idx );

def normalize( X ):
    X_norm = scale( X, axis = 0 )
    return( X_norm );
def drawline( P1, P2, *args, **kwargs ):
#    l = args.split(",")
    plt.plot( [ P1[0], P2[0] ], [ P1[1], P2[1] ], *args, **kwargs )
    return();

def pca( X, plotflag ):
    m, n = X.shape
    mu = np.mean( X, axis = 0 )
    U = np.zeros( n )
    S = np.zeros( n )
    Sigma = np.dot( X.T, X ) / m
    U, S, V = np.linalg.svd( Sigma )
    if plotflag == 1 :
        plt.scatter(X[:, 0], X[:, 1], s = 50, facecolors='none', edgecolors='black', zorder=20 )
        P1 = mu + 1.5 * S[0] * U[:,0]
        P2 = mu + 1.5 * S[1] * U[:,1]
#    plt.plot( [ mu[0], P1[0] ], [ mu[1], P1[1] ], 'g', linewidth=2 )
#    plt.plot( [ mu[0], P2[0] ], [ mu[1], P2[1] ], 'g', linewidth=2 )
        drawline( mu, P1, 'b', linewidth = 2 )
        drawline( mu, P2, 'b', linewidth = 2 )
        plt.axis( 'scaled' )
        plt.show()
    return(U, S);

def projectdata( X, U, K ):
    m, n = X.shape
    Z = np.zeros( (m, K) )
    for i in range( m ):
        for j in range ( K ):
            Z[i, j] = np.dot( X[i, :], U[:, j] )
    return( Z );

def recoverdata( Z, U, K ):
    m, n = Z.shape
    X_rec = np.zeros( (m, U.shape[0]) )
    for i in range( m ):
        for j in range( U.shape[0] ):
            X_rec[i,j] = np.dot( Z[i,:], U[j,0:K] )
    return(X_rec);

def runpca( file, K, plotflag ):
    X = loaddata( file )
    X = normalize( X )
    U, S = pca( X, 0 )
    Z = projectdata( X, U, K )
    X_rec = recoverdata( Z, U, K )
    if plotflag == 1 :
        plt.scatter(X[:, 0], X[:, 1], s = 50, facecolors='none', edgecolors='black', zorder=20 )
        plt.scatter(X_rec[:, 0], X_rec[:, 0], s=40, facecolors='none', edgecolors='r', zorder=20 )
        for i in range( X.shape[0] ):
            drawline( X[i,:], X_rec[i,:], 'b--', linewidth = 0.5 )
        plt.axis( 'scaled' )
        plt.show()
    return( X_rec );

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
            

    plt.figure(figsize=(7.5,7.5))
    plt.axis('off')
    plt.imshow(disp_array, aspect='auto', cmap='gray', origin="upper")
    plt.show()
    return();

#main program    
#kmeansclustering( 'ex7data2.mat', 3, 10, 1 )
#imagecompression( 'bird_small.mat', 16 )
#imagecompression( 'bird_small.png', 8 )
#imagecompression( 'bird2.png', 8 )
#runpca( 'ex7data1.mat', 1, 1 )
#X =loaddata( 'ex7faces.mat' )
#displaydata( X[0:100,:] )
#X_rec = runpca( 'ex7faces.mat', 100, 0 )
#displaydata( X_rec[0:100,:] )



#test
#D1 = spio.loadmat( 'ex7data1.mat' )
#D2 = spio.loadmat( 'ex7data2.mat' )
#X1 = np.require(D1['X'], dtype = np.float64, requirements = ['C'])
#X2 = np.require(D2['X'], dtype = np.float64, requirements = ['C'])
#c_init = np.array([[3, 3], [6, 2], [8, 5]])
#c_init = randinit( X2, 3 )
#idx = centassgn( X2, c_init )
#print( idx[0:3] )
#print( computecent( X2, c_init, idx ) )
#c, idx = runkmeans( X2, c_init, 10, 1 )
#print ( c, idx )