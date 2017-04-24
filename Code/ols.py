import numpy as np
from mySpark import RDD
import warmup
import time
from operator import add

def computeXY(rdd):    
    '''
    Compute the features times targets term
    needed by OLS. Return as a numpy array
    of size (41,)
    '''
    rdd_f=RDD.map(rdd, lambda x: np.append(x,[1]))
    rdd_p=RDD.map(rdd_f, lambda x: np.append(x[:-2],x[-1])*x[-2])
    rdd_res=RDD.reduce(rdd_p, lambda x,y: np.sum([x,y],axis=0))
    return rdd_res

def computeXX(rdd):
    '''
    Compute the outer product term
    needed by OLS. Return as a numpy array
    of size (41,41)
    '''    
    rdd_f=RDD.map(rdd, lambda x: np.append(x[:40],[1]))
    res1=RDD.reduce(RDD.map(rdd_f, lambda x: np.outer(x, np.transpose(x))), add)
    return np.asarray(res1)




    
def computeWeights(rdd):  
    '''
    Compute the linear regression weights.
    Return as a numpy array of shape (41,)
    '''

    inv=np.linalg.inv(computeXX(rdd))
    res=np.dot(inv,computeXY(rdd))
    return res

def computePredictions(w,rdd):  
    '''
    Compute predictions given by the input weight vector
    w. Return an RDD with one prediction per row.
    '''
    rdd_f=RDD.map(rdd, lambda x: np.append(x[:-1],[1]))
    res=RDD.map(rdd_f, lambda x: np.dot(w,x))
    return res
    
def computeError(w,rdd):
    '''
    Compute the MAE of the predictions.
    Return as a float.
    '''
    rdd_f=RDD.map(rdd, lambda x: np.append(x,[1]))
    res=RDD.map(rdd_f, lambda x: (x[-2], np.dot(np.append(x[:-2], x[-1]),w)))
    error=RDD.map(res, lambda x: abs(x[0]-x[1]))
    total=RDD.reduce(error, lambda x,y: x+y)
    mae=total/warmup.count(rdd)
    return mae
    

