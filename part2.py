from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering,KMeans
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value 
for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

model_sse_inertial={}
model_sse_manual={}
def fit_kmeans(X,k):
    sse_inertia=[]
    sse_man=[]
    for clusters_num in range(1,k+1):
      kmeans=KMeans(n_clusters=clusters_num)
      preds=kmeans.fit_predict(X)
      sse={}
      for iter,prediction in enumerate(preds):
        alp=(X[iter][0]-kmeans.cluster_centers_[prediction][0])**2 + (X[iter][1]-kmeans.cluster_centers_[prediction][1])**2
        try:
          sse[prediction]+=alp
        except:
          sse[prediction]=alp

      sse_inertia.append(kmeans.inertia_)
      sse_manual=0
      for i in sse:
        sse_manual+=sse[i]
      sse_man.append(sse_manual)
      model_sse_inertial[clusters_num]=kmeans.inertia_
      model_sse_manual[clusters_num]=sse_manual

    return sse_inertia,sse_man


def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    n_samples=20
    center_box=(-20,20)
    centers=5
    random_state=12
    x,label = datasets.make_blobs(n_samples=n_samples, centers=centers,center_box=center_box, random_state=random_state)
    #dct['bvv']=[bvv[0],bvv[1]]
    cord_1=x[0:,0:1]
    cord_2=x[0:,1:]

    


    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [cord_1,cord_2,label] #[np.zeros(0)]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
  
    X=np.concatenate([answers["2A: blob"][0],answers['2A: blob'][1]],axis=1)
    # print(X)
    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans
    # print('here')
    # print(dct)

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    sse_c=fit_kmeans(X,k=8)[1]
    # print(sse_c)
    #print([(i,sse_c[i-1]) for i in range(1,9)])
    sse_vs_k=[[x,y] for x,y in zip(range(1,9),sse_c)]
    # print(sse_vs_k)
    plt.plot(np.array(sse_vs_k)[:,1])
    plt.savefig("sse_vs_k_2c.png")
    # print(my_l)
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair

    dct = answers["2C: SSE plot"] =sse_vs_k#[[0.0, 100.0]]
    # print('Here we are')
    # print(dct)

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    sse_d=fit_kmeans(X,k=8)[0]
    sse_vs_k=[[x,y] for x,y in zip(range(1,9),sse_d)]
    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = sse_vs_k
    # print('Inertia')
    # print(dct)
    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
