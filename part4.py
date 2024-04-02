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
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
import myplots as myplt
from matplotlib.backends.backend_pdf import PdfPages


"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge 
clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data,linkage_type,num_of_clusters):
    model = AgglomerativeClustering(n_clusters=num_of_clusters, linkage=linkage_type)
    ss=StandardScaler()
    train=ss.fit_transform(data[0])
    model.fit(train,data[1])
    labels = model.labels_
    return labels


def fit_modified(data,distance_threshold,linkage_type):
    model = AgglomerativeClustering(n_clusters=None,distance_threshold=distance_threshold, linkage=linkage_type)
    ss=StandardScaler()
    train=ss.fit_transform(data[0])
    model.fit(train,data[1])
    labels = model.labels_
    return labels
    # return None


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster 
    (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an 
    AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. 
    Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}
    n_samples = 500
    seed = 42
    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    dct['nc']=[nc[0],nc[1]]

    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    dct['nm']=[nm[0],nm[1]]

    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    dct['bvv']=[bvv[0],bvv[1]]

    X, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)
    dct['add']=[add[0],add[1]]

    b = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    #print(b[0].shape,b[1].shape)
    dct['b']=[b[0],b[1]]
    


    # dct value:  the `fit_hierarchical_cluster` function

    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, 
    centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, 
    are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    hierarchical_dict_plotting={}
    for dataset_i in answers['4A: datasets'].keys():
        acc=[]
        dataset_cluster={}
        for linkage_ in ['single','complete','ward','average']:
            preds=dct((answers['4A: datasets'][dataset_i][0],answers['4A: datasets'][dataset_i][1]),linkage_,2)
            dataset_cluster[linkage_]=preds
        acc.append((answers['4A: datasets'][dataset_i][0],answers['4A: datasets'][dataset_i][1]))
        acc.append(dataset_cluster)
        hierarchical_dict_plotting[dataset_i]=acc
 
    myplt.plot_part1C(hierarchical_dict_plotting,'Part4_B.jpg')
    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
    dct = answers["4C: modified function"] = fit_modified
  

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()
    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
