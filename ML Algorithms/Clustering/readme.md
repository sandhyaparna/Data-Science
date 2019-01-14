### Links
https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/ <br/>
https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right/ <br/>
https://www.analyticsvidhya.com/blog/2013/11/getting-clustering-right-part-ii/ <br/>


### Overview
Segmenting is the process of putting customers into groups based on similarities, and clustering is the process of finding similarities in customers so that they can be grouped, and therefore segmented. <br/>
Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters. <br/>
* Connectivity models: start with classifying all data points into separate clusters & then aggregating them as the distance decreases (or) all data points are classified as a single cluster and then partitioned as the distance increases. Lacks scalability for handling big datasets. (Hierarchical)
* Centroid models: Iterative clustering algorithms in which the notion of similarity is derived by the closeness of a data point to the centroid of the clusters. (K-Means)
* Distribution models: Based on the notion of how probable is it that all data points in the cluster belong to the same distribution (For example: Normal, Gaussian). These models often suffer from overfitting. (Expectation-maximization algorithm )
* Density Models: These models search the data space for areas of varied density of data points in the data space. It isolates various different density regions and assign the data points within these regions in the same cluster. Popular examples of density models are DBSCAN and OPTICS.
* Hierarchical cannot handle large data sets as it is time consuming
* K-Means is more suitable for large data sets


### Clustering Algos
* K-Means - Representative based clustering
* Mean-Shift
* Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
* Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
* Agglomerative Hierarchical
* Mini-Batch
* Affinity Propagation
* Spectral Clustering - Spectral and graph clustering
* Ward
* Agglomerative Clustering - Hierarchical clustering
* Birch

### Categorical Clustering Algos
* K-Modes
* Squeezer
* LIMBO
* GAClast
* Cobweb Algo
* STIRR, ROCK, CLICK
* CACTUS, COOLCAT, CLOPE

###
Segmenting is the process of putting customers into groups based on similarities, and clustering is the process of finding similarities in customers so that they can be grouped, and therefore segmented
Relative clustering validation: find optimal number of clusters (eg- elbow method - within sum of squares minimization)
External Clustering validation: Comparing it externally provided class values - supervised
Internal clustering validation: silhouette 
k-modes, K-protypes can be used for mixed data
Clustering, Data Segmentation - Similar group of Items based on features - Unsupervised learning
Proximity measures for effective clustering
How to judge quality of clusters - silhouette in R
Helps in trends & pattern discovery, classification an outlier analysis
Applications: Image Processing-vector quantization, collaborative filtering, recommendation systems(Market Basket Analysis) or customer segmentation, social network analysis, clustering audio/video clips
Partioning Methods - KMeans, K-Medians, K-Medoids, Kernel K-means
Hierarchical - BIRCH, CURE, CHAMELEON,
Density based - DBSCAN, OPTICS
Grid-based - CLIQUE, STING, DENCLUE
cluster is a collection of data objects which are similar to one another within the same group and dissimilar to the objects in other groups
Challenges: How to want to partition data - multi level or single level; How to we seprerate clusters from one another - exclusive or non-exclusive; similarity measure - Distance-based vs. connectivity eg.density; clustering space - 2 dim or high-dim; Quality - diff types of data, arbitrary shape of data, ability to deal with noisy data; Scalability - All data or sample; constraints of user -domain knowledge; Interpretability & usability
Technique Centered - Distance based methods,
clustering is applied on - Numerical, categorical, text, multimedia, time-series, sequences, networked, uncertain data
Clustering Methodologies -  1) Distance-based methods: a) Partioning algorithms: k-means, (k-medians, k-medoids are better at handling noisy data and outliers compared to k-means) b) Hierarchical algorithms: Agglomerative vs Divisive methods 2) Density-based & grid-based methods: a) Density-based: Data space is explored at high-level of granularity & then post-processing to put together dense regions into an aarbitary shape b) Grid-based: Individual regions of data space are formed into a gri-like structure 3) Probabilistic & generative models: Model params are estimated with the Expectation-Maximization algorithm
High-dimensional clustering - a)subspace clustering: find clusters on various sub spaces - bottom-up or top-down, correlation based methods b) Dimensionality Reduction - vertical form of clustering-columns clustering, probabilistic latent semantic indexing, spectral clustering
Text data clustering - combination of k-means & agglomerative; topic modeling ; co-clustering
Sequence data clusteering uses Suffix tree, generative model eg. Hidden Markov Model
Stream data requires single pass algorithm eg. Micro clustering
Good clustering method will produce clusters with high intra-class similarity and low inter-class similarity
Numerical Data - Distance measures - Euclidean dist, Minkowski dist, Manhattan Dist, Supremum dist
Binary Data - Proximity measure for Binary Attributes - Symmetric binary(Dist measure), assymetric binary(DIst measure), Jaccard coef(Similarity Measure),
Proximity measures for categorical attributes - simple matching, use a large number of binary attributes
Attributes of mixed type - Use a weighted fromula to combine their effects - (numeric=Normalized dist),B
Text clustering - Performed using Cosine similarity of Two vectors - Freq of bag of words to determine similarity between two documents
Variance - How widely individuals in a group vary or expected value of sq deviation from the mean
Covarinace - Measure of how changes in one variable are associated with changes in a second variables. Measures the degree to which 2 variables are linearly associated
Correlation - Scaled version of covariance that takes on values in [-1,1]
If 2 variables are independent, the covariance is 0. But, having a covariance of 0 doesnt imply the variables are independent (there can be non-linear relationship)
Partioning Methods - the objective function, Sum of squared errors must be minimized
k-means can only find the local minimum of SSE. Implies each k - there is a minimum SSE that can be obtained (Global minimum is obtained by iteration of diff values of k)
K-means finally converges after iterations
k-means is applicable only to objects in a continuous n-dimensional space (k-modes is used for categorical data)
Distance measures used for Numerical data - Euclidean dist, Minkowski dist, Manhattan Dist, Supremum dist, Cosine Similarity, Correlation Coefficient
k-means is not suitable to discover clusters with non-convex shapes. For such density-based clustering, kernel k-means etc are used
k-Means++ is used fro better initialization of k - first centroid is selected at random, the next centroid selected is the one that is farthest from the currently selected one and continues util k centroids are obtained
Some different intializations of k may generate the same clustering result
k-means is sensitive to outliers
k-medoids : select intial k medoids randomly. Repeat re-assignment, swap medoid m with Oi, if it imporves the clustering quality until convergence criterion is satisfied
k-medians : the (medians) center for each cluster are computed on dimeansion basis and the new center may not be a real data point
k-modes: For categorical data
k-Prototype - An integration of k-means and k-modes is used when there are both Numerical and Categorical
Kernel k-means is used to detect non-convex clusters (eg: 2 ellipses)
Spectral clustering is a variant of Kernel k-means clustering
Agglomerative Clustering - Bottom-up - Continuously merge nodes that have the least dissimilarity Single link(Nearest neighbor), Average link, Complete link(Diameter), Centroid link, Ward's Criterion
Ward's Criterion - The inc in the value of the SSE criterion for the clustering obtained by merging 2 clusters
Divisive Clustering - Top-Down may use ward's criterion to chase for greater reduction in the difference in the SSE criterion as a result of a split Gini-index can be used to categorical data Noise can be handled by determing a threshold for termination criterion
Weaknesses of Hierarchical clustering - can never undo what was done previously
other Hierarchical clustering - BIRCH, CURE, CHAMELEON
BIRCH - Idea of macro and micro clustering using clustering feature tree and incrementally adjust the quality of sub-clusters
CURE - Represent a cluster using a set of well-scattered points
CHAMELEON - Use graph partitioning methods on k-NN graph of the data
BIRCH(Balanced Iterative Reducing & Clustering using Hierarchies) - A multiphase clustering algorithm Low-level micro-clustering - Reduce complexity & increase scalability High-level Macro-clustering - Leave enough flexibility fro high-level clustering
CURE(Clustering using Representatives) - Represent a cluster using a set of well-scattered representative points. More robust to outliers
CHAMELEON - Measures similarity based on a dynamic model. Two clusters are merged only if the interconnectivity & closeness between two clusters are high relative to the internal interconnectivity of the clusters and closeness of items within the clusters Merges only those pairs of sub-clusters whose RI and RC are above some user specified thresholds Once a k-NN graph is constructed, it is partitioned into small graphlets, which are then merged back together to create clusters Can generate high quality clusters even with complex data
Probabilistic Hier Clust - Use probabilistic models to measure distance between clusters
DBSCAN -Density based Clustering based on density(a local cluster criterion) such as density connected points Major features are: discover clusters of arbitrary shape, handle noise, one scan, need density parameters as termination condition(is senitive to the setting of params)
OPTICS - Ordering points to identify clustering structure - uses core distance and reachability distance
Clustering validation & assessmnet : Unsupervised and dont have expert to judge clustering evaluation - evaluating the goodness of the clustering Clustering stability - to understand the sensitivity of the clustering result to various algorithm parameters eg-no of clusters clustering tendency - Assess the suitability of clustering - whether the data has any inherent grouping structure
Measure cluster quality - 3 categorization of measure - external, internal and relative  External - Supervised, employ criteria not inherent to the data set - Compare a clustering against prior or expert-specified knowledge jusing certain clustering quality measure Internal - Silhouette coeff - Evaluate the goodness of a clustering by considering how well the clusters are seperated and how compact the clusters are Relative - Directly compare different clusterings, usually those obtained via different parameter settings for the same algorithm
Asses the suitability of clusters -  Spatial Histogram - Contrast the histogram of data with that generated from random samples (Kullback-Leibler divergence value) Distance Distribution - Compare the pairwise point distance from the data with those from the randomly generated samples Hopkins Statistic - A sparse sampling test for spatial randomness
Squeezer R code
Is there a way to find optimal clusters for K-modes similar to the one like k-means






Centroid Based, Connectivity Based, Density Based, Probabilistic





