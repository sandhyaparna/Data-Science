KMeans, Low pass filter https://www.datascience.com/blog/python-anomaly-detection <br/>
PCA + Mahalanobis distance, NN https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7 <br/>
Above artcile Part2 https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770 <br/>

### Detection
* Lower & Upper limit
* For identifying anomalies with 1 or 2 variables, data visualization is a good starting point
* PCA + Mahalanobis distance https://www.statisticshowto.datasciencecentral.com/mahalanobis-distance/ <br/>
  * Mahalanobis distance (MD) is the distance between two points in multivariate space. (It can calculate dist even for correlated vars).
  * The Mahalanobis distance measures distance relative to the centroid — a base or central point which can be thought of as an overall mean for multivariate data. The centroid is a point in multivariate space where all means from all variables intersect. The larger the MD, the further away from the centroid the data point is.
  * d (Mahalanobis) = [(xB – xA)T * C -1 * (xB – xA)]0.5
  * A major issue with the MD is that the inverse of the correlation matrix is needed for the calculations. This can’t be calculated if the variables are highly correlated - Hence, PCA helps as it produces uncorrelated variables
  * PCA is a linear mapping of the data into a lower dimensional space such that variance is maximized/increased by bringing the data into low dimensional space. Loses a little accuracy
  * Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables. These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components.
  * Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.

 <br/>
luminol - https://github.com/linkedin/luminol <br/>
Moving Avergae Based - https://www.datascience.com/blog/python-anomaly-detection <br/>
https://github.com/rob-med/awesome-TS-anomaly-detection <br/>
Surveillance Algos - https://www.ml.cmu.edu/research/dap-papers/das_kdd2.pdf <br/>
https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-10-74 <br/>

##### Why use Surveillance algorithms
For each possible organism we have a corresponding time series. A particular disease such as an influenza outbreak, will affect the count of multiple syndromes. In this case, we need to simultaneously consider all the variables to detect the presence of an anomaly. To combine information from multiple time series we examine a novel technique which is simple but powerful. Composite time series are constructed by simple addition and subtraction of the individual time series. We search through all possible composite time series for an anomaly. Using just simple arithmetic operations like addition and subtraction provides an easy physical interpretation of the composite series. It is also able to detect anomalies sooner than other traditional methods
* Vector Auto Regression
* Vector Moving Average
* Hotelling T2 Test

* EARS algo, Farrington flexible are applied on weekly data
* CUSUM - Detect a shift in the mean of a process. CUSUM maintains a cumulative sum of deviations from a reference value r.
* Modified CUSUM - Model non-stationary time series variables
* Multivariate CUSUM






