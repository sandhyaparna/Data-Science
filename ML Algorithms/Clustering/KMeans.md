### Links
https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/
https://www.saedsayad.com/clustering_kmeans.htm


### Overview
K-Means clustering intends to partition n objects into k clusters in which each object belongs to the cluster with the nearest mean.
The objective of K-Means clustering is to minimize total intra-cluster variance, or, the squared error function.

##### Algorithm
* Clusters the data into k groups where k  is predefined.
* Select k points at random as cluster centers.
* Assign objects to their closest cluster center according to the Euclidean distance function.
* When all objects have been assigned, recalculate the cluster centers (or) centroid
* Repeat steps 3 and 4 until the same points are assigned to each cluster in consecutive rounds (or) until the centroids no longer move


