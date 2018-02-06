# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:13:54 2018

@author: zhenl
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as spio
import re

def loaddata( name, plotflag ):
    D = spio.loadmat( name )
    var = spio.whosmat( name )
    X1 = np.require(D[var[0][0]], dtype = np.float64, requirements=['C'])
    X2 = np.require(D[var[1][0]], dtype = np.float64, requirements=['C'])
    if plotflag == 1 :
        plt.figure(figsize=(7.5,7.5))
        plt.axis('off')
        plt.imshow(Y, aspect='auto', cmap = "OrRd", origin = "upper")
        plt.xlabel( 'Users' )
        plt.ylabel( 'Movies' )
        plt.show()
    return( X1, X2 );
    
def coficostfunc( params, Y, R, n_movies, n_features, n_users, lambda_ ):
    X, Theta = unfoldparams( n_movies, n_features, n_users, params );
    J = ( np.sum( ( np.dot( X, Theta.T ) * R - Y * R ) ** 2 ) /2 + 
         np.sum( Theta**2 ) * lambda_ / 2 + np.sum( X **2 ) * lambda_ / 2 )
    grad_X = ( np.dot( ( np.dot( X, Theta.T ) * R - Y * R ), Theta ) + 
              lambda_ * X )
    grad_Theta = ( np.dot( ( np.dot( X, Theta.T ) * R - Y * R ).T, X ) + 
                  lambda_ * Theta )
    grad_params = np.concatenate( 
            (grad_X.flatten(), grad_Theta.flatten()), axis=0 )
    return( J, grad_params );

def addmyratings( Y, R, movie_list ):
    flag = int( input( 'Add ratings: default <0> or customize <1>? --> ' ) )
    my_ratings = np.zeros( (len( movie_list),1) )
    if flag == 0:
        my_ratings[0] = 4
        my_ratings[97] = 2
        my_ratings[6] = 3
        my_ratings[11] = 5
        my_ratings[53] = 4
        my_ratings[63] = 5
        my_ratings[65] = 3
        my_ratings[68] = 5
        my_ratings[182] = 4
        my_ratings[225] = 5
        my_ratings[354] = 5
        print( 'added default ratings for selected movies.' )
    elif flag == 1:        
        while True:
            id_= int(input( 'movie id(1-1682) <enter 0 to exit> --> ' ))
            if id_ == 0:
                break;
            rating = int(input( 'movie rating(1-5) --> ' ))
            my_ratings[ id_ - 1 ] = rating
        print( 'added user-defined ratings.' )            
    Y = np.c_[ my_ratings, Y ]
    R = np.c_[ ( my_ratings != 0 ).astype(int), R ]
    return( Y, R, my_ratings )
    
def loadmovielist( filename ):
    mov_list = []
    with open( filename, encoding = 'ISO-8859-1' ) as fp:
        for line in fp:
            line = line.rstrip()
            line = re.findall( r'[^\s]+', line )
            mov_list.append( ' '.join(line[1::]) )
    return( mov_list ); 
    
def normalizeratings( Y, R ):
    Y_sum = np.sum( Y, axis = 1 )
    R_sum = np.sum( R, axis = 1 )
    Y_mean = Y_sum / R_sum
    Y_norm = np.zeros( Y.shape )
    for i in range( Y.shape[0] ):        
        j, = np.where( R[i,:] == 1 )
        Y_norm[i,j] = Y[i,j] - Y_mean[i]
    return( Y_norm, Y_mean ); 

def initparams( n_movies, n_features, n_users ):
    X = np.random.randn( n_movies, n_features )
    Theta = np.random.randn( n_users, n_features )
    init_params = np.concatenate( (X.flatten(), Theta.flatten()), axis=0 )
    return( init_params );
    
def unfoldparams( n_movies, n_features, n_users, theta ):
    X = theta[0:n_movies * n_features].reshape( n_movies, n_features )
    Theta = theta[n_movies * n_features::].reshape( n_users, n_features )
    return( X, Theta );
    
if __name__ == "__main__" :
    Y, R = loaddata( 'ex8_movies.mat', 0 )
    print ("avg rating for Toy Story (1995) is %.2f / 5." % np.mean(
            Y[0,:][R[0,:].astype(bool)]) )
    mov_list = loadmovielist( 'movie_ids.txt' )
#add my own ratings for movies in this list
    Y, R, myratings = addmyratings( Y, R, mov_list );
    n_movies, n_users = Y.shape
    Y_norm, Y_mean = normalizeratings( Y, R );
    n_features = 10
    init_params = initparams( n_movies, n_features, n_users );
    
    theta_ = scipy.optimize.minimize(
        fun = coficostfunc,
        x0 = init_params,
        args = (Y_norm, R, n_movies, n_features, n_users, 10 ),
        method = 'CG',
        jac = True,
        options = {
        'maxiter': 100,
        'disp': False,}).x
    
    X, Theta = unfoldparams( n_movies, n_features, n_users, theta_ )    
    my_predictions = np.dot( X, Theta.T )[:,0] + Y_mean
    idx = my_predictions.argsort()[::-1][:20]
    print ( 'Recommendations for you: ' )
    for j in idx :
        print ( '<Rating: %.1f>' % my_predictions[j], mov_list[j], Y_mean[j], np.sum( R, axis = 1 )[j] )
    
#test
#    X, Theta = loaddata( 'ex8_movieParams.mat', 0 )
#    num_users = 4; num_movies = 5; num_features = 3;
#    X = X[0:num_movies, 0:num_features];
#    Theta = Theta[0:num_users, 0:num_features];
#    Y = Y[0:num_movies, 0:num_users];
#    R = R[0:num_movies, 0:num_users];
#    print ( coficostfunc( X, Theta, Y, R, 1.5 ))