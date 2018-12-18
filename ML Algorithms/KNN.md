### Links
http://www.saedsayad.com/k_nearest_neighbors.htm
https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/


### Overview
* KNN is non-parametric learning algorithm, which means that it doesn't assume anything about the underlying data. 
* Lazy learning is a learning method in which generalization of the training data is delayed until a query is made to the system, as opposed to in eager learning, where the system tries to generalize the training data before receiving queries.
* kNN algorithm uses the entire dataset as the training set, rather than splitting the dataset into a training set and test set.
* When an outcome is required for a new data instance, the KNN algorithm goes through the entire dataset to find the k-nearest instances to the new instance, or the k number of instances most similar to the new record, and then outputs the mean of the outcomes (for a regression problem) or the mode (most frequent class) for a classification problem. The value of k is user-specified.
* The training phase of the algorithm consists only of storing the feature vectors and class labels of the training samples. The similarity between instances is calculated using measures such as Euclidean distance and Hamming distance. In the testing phase, a test point is classified by assigning the label which are most frequent among the k training samples nearest to that query point – hence higher computation.
* The kNN task can be broken down into writing 3 primary functions: 
  * Calculate the distance between any two points (All points)
  * Find the nearest neighbours based on these pairwise distances
  * Majority vote on a class labels based on the nearest neighbour list 
* Feature scaling has to be performed before implementing the algorithm 
* Optimal k is determined by checking graph of Validation/Test error vs k
* k-NN algorithm can be used for imputing missing value of both categorical and continuous variables.
* Bias increases as you increase k implies Variance dec (Large k = Simple Model)

### Measures of Similarity
* Euclidean distance = sqrt((a1-b1)^2 + (a2-b2)^2) where (a1, a2) and (b1, b2) are two points.
* Manhattan 
* Minkowski
* Tanimoto
* Jaccard
* Mahalanobis
* Hamming Distance - Categorical Vars

### Disadvantages
* The KNN algorithm doesn't work well with high dimensional data because with large number of dimensions, it becomes difficult for the algorithm to calculate distance in each dimension.
* The KNN algorithm has a high prediction cost for large datasets. This is because in large datasets the cost of calculating distance between new point and each existing point becomes higher.
* Finally, the KNN algorithm doesn't work well with categorical features since it is difficult to find the distance between dimensions with categorical features.



