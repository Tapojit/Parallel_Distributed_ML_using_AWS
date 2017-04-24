from pyspark import SparkContext, SparkConf, RDD
import numpy as np
import time
from pyspark.mllib.regression import RidgeRegressionWithSGD, LabeledPoint


#============
#Replace the code below with your code


 

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
    
    train_mod=rdd_train.map(lambda x: LabeledPoint(x[-1], x[:-1]))
    model=RidgeRegressionWithSGD.train(train_mod)
    
    tuples=RDD.map(rdd_test, lambda x: (x[-1], model.predict(x[:-1])))
    error_RDD=RDD.map(tuples, lambda x: abs(x[0]-x[1]))
    total=RDD.reduce(error_RDD, lambda x,y: x+y)
    err = total/rdd_test.count()

    
    this_time =  time.time()-start
    print "\n\n\nCores %d: MAE: %.4f Time: %.2f"%(i, err,this_time)
    times.append([i,this_time])

    sc.stop()

print "\n\n\n\n\n"
print times
