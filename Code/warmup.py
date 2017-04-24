import numpy as np
from mySpark import RDD



def count(rdd):
    '''
    Computes the number of rows in rdd.
    Returns the answer as a float.
    '''
    return float(np.shape(RDD.collect(rdd))[0])

def mean(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (D,)
    Returns the sample mean of each column of rdd as a numpy array of shape (D,)
    '''
    return np.mean(RDD.collect(rdd), axis=0)

def std(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (D,)
    Returns the sample standard deviation of 
    each column of rdd as a numpy array of shape (D,)
    '''

    return np.std(RDD.collect(rdd), axis=0)/np.sqrt(count(rdd))
   

def dot(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (2,)
    Returns the inner (dot) product between the columns as a float.
    '''  
    row1=RDD.collect(RDD.map(rdd, lambda x: x[0]))
    row2=RDD.collect(RDD.map(rdd, lambda x: x[1]))
    return np.inner(row1,row2)

def corr(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (2,)
    Returns the sample Pearson's correlation between the columns as a float.
    '''  
    row1=RDD.collect(RDD.map(rdd, lambda x: x[0]))
    row2=RDD.collect(RDD.map(rdd, lambda x: x[1]))
    return np.corrcoef(row1, row2)[1,0]
    
def distribution(rdd):
    '''
    Assumes each row of rdd is a numpy array of shape (1,)
    and that the values in rdd are whole numbers in [0,K] for some K.
    Returns the empirical distribution of the values in rdd
    as an array of shape (K+1,)
    '''
    count2=np.bincount(RDD.collect(rdd))
    a=np.nonzero(count2)[0]
    freq=[row[1] for row in np.vstack((a,count2[a])).T]
    np.array(freq)/np.sum(freq).astype(float)
    return  np.array(freq)/np.sum(freq).astype(float)










