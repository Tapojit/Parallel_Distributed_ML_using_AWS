from mySpark import RDD
from mySpark import SparkContext
import numpy as np
import warmup
import ols
import matplotlib.pyplot as plt


SC  = SparkContext()

#Create data for Question 1
np.random.seed(0)
X    = np.hstack((np.random.randint(0,5,(100,1)),np.random.rand(100,1),np.random.randn(100,1)))
rdd1 = SC.parallelize(X) 



print "Question 1:"
 
print "1.1 Count:", warmup.count(rdd1)
 
print "1.2 Mean:", warmup.mean(rdd1)
 
print "1.3 Std:", warmup.std(rdd1)
 
print "1.4 Dot:", warmup.dot(rdd1)
 
print "1.5 Corr:", warmup.corr(rdd1)
  
rdd2 = rdd1.map(lambda x: x[0])

print "1.6 Distribution:", warmup.distribution(rdd2)
# 
# 
#Load data for question 2 assuming you are running local_run_me.py 
#from the Submission/Code directory. Do not change path to data.
 
path = '../../Data/Movielens/'
train = np.loadtxt(path + 'train_data.csv', delimiter = ',')
train[:,-2] = np.random.randn(train.shape[0])/100.0
test = np.loadtxt(path + 'test_data.csv', delimiter = ',')
rdd_tr = SC.parallelize(train)
rdd_te = SC.parallelize(test)
# 
print "Question 2:"
 
print "2.1 computeXY:", ols.computeXY(rdd_tr)
# 
print "2.2 computeXX:", ols.computeXX(rdd_tr)
# 
w=ols.computeWeights(rdd_tr)
 
print "2.3 computeWeights:", w 
# 
print "2.4 computePredictions:", ols.computePredictions(w,rdd_te)
# # 
print "2.5 computeError:", ols.computeError(w,rdd_te)
# #  

#Array of tuples for OLS   
worker_time=[[1, 510.902055978775], [2, 265.4460961818695], [3, 173.28980493545532], [4, 135.55232405662537], [5, 145.72118091583252], [6, 161.2725579738617], [7, 170.14717197418213], [8, 169.82594799995422]]

def line_plot(x_lab, y_lab, x, y, title):
    values =y
    inds   =x
    #Plot a line graph
    plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.plot(inds,values,'or-', linewidth=3) #Plot the first series in red with circle marker
    
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel(y_lab) #Y-axis label
    plt.xlabel(x_lab) #X-axis label
    plt.title(title) #Plot title
    plt.xlim(0, 10) #set x axis range
    plt.ylim(0,600) #Set yaxis range
    
    #Make sure labels and titles are inside plot area
    plt.tight_layout()
    
    #Save the chart
    plt.savefig("../Figures/"+title+".png")
    
    print "Line graph image generated"
time_arr=[worker_time[a][1] for a in range(len(worker_time))]
# line_plot("workers", "Run Time", range(1, 9), time_arr, "Run Time vs Workers")

#Array of tuples for Ridge Regression with SGD
worker_time2=[[1, 347.75902581214905], [2, 191.83908796310425], [3, 144.88778495788574], [4, 121.55717206001282], [5, 140.06287217140198], [6, 137.843416929245], [7, 149.11939096450806], [8, 141.34602284431458]]
time_arr2=[worker_time2[a][1] for a in range(len(worker_time2))]
#line_plot("Workers", "Run Time", range(1, 9), time_arr2, "Run Time vs Workers-3.3")














