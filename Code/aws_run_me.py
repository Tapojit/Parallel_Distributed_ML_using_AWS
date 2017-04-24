from pyspark import SparkContext, SparkConf, RDD
import numpy as np
import time
from operator import add

#============
#Replace the code below with your code


# def computeWeights(rdd):
#    return np.random.randn(41) 
#    
# def computeError(w,rdd_test):
#    return 0  
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
    mae=total/rdd.count()
    return mae

#============

#Convert rdd rows to numpy arrays
def numpyify(rdd):
    return rdd.map(lambda x: np.array(map(lambda y: float(y),x.split(","))))

#sc     = spark.sparkContext
master = "yarn"
times=[]


#Flush yarn defaul context
sc = SparkContext(master, "aws_run_me")
sc.stop()

for i in [1,2,3,4,5,6,7,8]:

    conf = SparkConf().set("spark.executor.instances",i).set("spark.executor.cores",1).set("spark.executor.memory","2G").set("spark.dynamicAllocation.enabled","false")
    sc = SparkContext(master, "aws_run_me", conf=conf)
    sc.setLogLevel("ERROR")

    start=time.time()

    rdd_test = numpyify(sc.textFile("s3://589hw03/test_data_ec2.csv"))        
    rdd_train = numpyify(sc.textFile("s3://589hw03/train_data_ec2.csv"))

    w = computeWeights(rdd_train)
    err = computeError(w,rdd_test)
    
    this_time =  time.time()-start
    print "\n\n\nCores %d: MAE: %.4f Time: %.2f"%(i, err,this_time)
    times.append([i,this_time])

    sc.stop()

print "\n\n\n\n\n"
print times
