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







Centroid Based, Connectivity Based, Density Based, Probabilistic





