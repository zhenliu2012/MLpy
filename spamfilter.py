#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:29:30 2018

@author: zhenliu
"""
import pandas as pd
from sklearn import svm
import numpy as np
import scipy.io as spio
from bs4 import BeautifulSoup
import re
from nltk.stem.porter import PorterStemmer
#f = open('emailSample1.txt','r')
#message = f.read()
#print(message)

#for link in soup.find_all('a'):
#    print (link.get('href'))    

def reademail( filename ):
    f = open( filename, 'r' )
    message = f.read()
    print( 'Original email content:' )
    print( message )
    return( message )

def processemail( message, vocabfile ):
# strip html tags, leaving only the contents
    soup = BeautifulSoup( message, "lxml" )
    message = soup.get_text()
# change text to lower case    
    message = message.lower()
# replace http urls with 'httpaddr'
    message = re.sub( r'https?://[^\s]*', 'httpaddr', message )
    ''' [^\s] means any character but not a whitespace character '''
# replace numbers with 'number'
    message = re.sub( r'[0-9]+', 'number', message )
# replace email address with 'emailaddr'
    message = re.sub( r'[^\s]+@[^\s]+', 'emailaddr', message )
# replace dollar sign with 'dollar'
    message = re.sub( r'[$]+', 'dollar', message )
# remove punctuation
#    message = re.split( r'\W+', message )
# NOTE: re.split returns empty str at the beginning and the end of the list
    
    message = re.findall( r'[^\W]+', message )
# stemming
    message = [ PorterStemmer().stem(word) for word in message ]
#    print ( ' '.join( message ) )
# extract features  

    df = pd.read_table( vocabfile, delim_whitespace=True, names=('N', 'W'))
    n = len(df.index)
    vec = np.zeros((1,n), dtype = np.int )
# mapping if needed
    message_new = []
    for word in message:
        df_tar = df[df['W'] == word]
#    print(df.loc[df['W'] == word]['W'].values)
        if df_tar.empty:
#            print ( "'", word, "'", 'does not exist in the vocab list!' )
            continue
        else:
            str_ = df_tar.loc[df['W'] == word]['N'].values[0]
#        print ( str_ )
            message_new.append(str_)
            vec[0, str_ - 1] = 1
#    print (message_new)
#    print ( vec )
    return(vec);
    
def predictspam( filename, clf ):
    message = reademail( filename )
    vec = processemail( message, 'vocab.txt' )
#   print( len(vec), np.count_nonzero(vec) )
    Z = clf.predict( vec )
    if Z == 1:
        print ( '>>>>>>>>>>!!!SPAM!!!<<<<<<<<<<\n' )
    else:
        print ( '>>>>>NOT spam<<<<<\n' )
    return( Z );
    
def toppredictors( linearclf ):
#weights -- top predictors    
    w = (linearclf.coef_.flatten()).argsort()[::-1][:20] + 1
    df = pd.read_table( 'vocab.txt', delim_whitespace=True, names=('N', 'W'))
    for k in w:
        print( df.loc[df['N'] == k]['W'].values[0] )
    return();

if __name__ == "__main__" :

#main program:

#train linear classifiers using SVMs
    D1 = spio.loadmat('spamTrain.mat') #training set
    D2 = spio.loadmat('spamTest.mat') #val set
    X1 = np.require(D1['X'], dtype = np.int, requirements=['C'])
    y1 = np.require(D1['y'].flatten(), dtype = np.int)
    Xval = np.require(D2['Xtest'], dtype = np.int, requirements=['C'])
    yval = np.require(D2['ytest'].flatten(), dtype = np.int)

    clf = svm.SVC(kernel='linear', C = 0.1)
    clf.fit(X1, y1);
    toppredictors( clf );


#    Z = clf.predict( X1 )
#    print( 'accuracy <training set>:', np.mean( Z == y1 ) )
#    Z = clf.predict( Xval )
#    print( 'accuracy <validation set>:', np.mean( Z == yval ) )

#predict spam(1) or non-spam(0) emails using this clf
    predictspam( 'emailSample1.txt', clf )
#    predictspam( 'emailSample2.txt', clf )
