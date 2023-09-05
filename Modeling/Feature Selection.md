### Links
Genetic Algorithm https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29 <br/>
https://topepo.github.io/caret/feature-selection-using-genetic-algorithms.html <br/>
Dimension reduction techniques https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/ <br/>
Filter, Wrapper Methods https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/ <br/>

### Reasons to include fewer predictors over many
* Redundancy/Irrelevance: Remove non-redundant predictor variables. Pareto principle suggests that 80% of what happens can be explained using 20%. Irrelevant or partially relevant features can negatively impact model performance.
* Garbage In = Garbage Out - So, we are trying to avoid it
* Over-fitting: The data models with large number of predictors (also referred to as complex models) often suffer from the problem of overfitting, in which case the data model performs great on training data, but performs poorly on test data.
* Improves Accuracy/Performance (Less Noise)
* Reduces Training time
* Understandability/Interpretability: Fewer predictors are way easier to understand and explain

### Feature reduction techniques: Numeric & Categorical - Domain dependent
https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0 </br>
* Missing value ratio
* Low Variance filter - Variance Threshold
* Remove highly correlated variables among the independent variables
* Regularization - Lasso
* Boruta Feature Selection Algorithm - https://medium.com/swlh/feature-importance-hows-and-why-s-3678ede1e58f
  * Boruta tries to find all features carrying useful information rather than a compact subset of features that give a minimal error.
  * Feature Permutation means, for one feature, we randomly shuffle this feature’s order and breaks the relationship between the feature and the target variable. The basic intuition for Feature Permutation is:
We assume the shuffled feature should have less importance than a good feature, so if one feature has less feature importance than the shuffled feature, then that feature is a candidate to be removed.
  * https://towardsdatascience.com/simple-example-using-boruta-feature-selection-in-python-8b96925d5d7a
  * https://towardsdatascience.com/feature-selection-with-boruta-in-python-676e3877e596
* Forward slection, Backward Selection, Stepwise Selection 
  *  Stepwise selection alternates between forward and backward, bringing in and removing variables that meet the criteria for entry or removal, until a stable set of variables is attained.
* Recursive Feature Elimination (concept is same as Backward selection) https://machinelearningmastery.com/rfe-feature-selection-in-python/
  * RFE works by searching for a subset of features by starting with all features in the training dataset and successfully removing features until the desired number remains.
  * There are 3 main parameters to sklearn’s RFE method.
    * estimator — a machine learning model with a .fit method
    * n_features_to_select — the number of features that will be kept
    * steps — how many features are dropped every time RFE reduces features
* Features with 0 importance in a tree based model
* Chi-sq for Categorical data - SelectKBest based on chi-sq score in python 
* F-test capture linear relationship well (similar to ANOVA but this dont require normality assumption)
  * Scikit learn provides the Selecting K best features using F-Test.
  * sklearn.feature_selection.f_regression
  * sklearn.feature_selection.f_classif
* Mutual info: Mutual Information between two variables measures the dependence of one variable on another. If X and Y are two variables, and
  * If X and Y are independent, then no information about Y can be obtained by knowing X or vice versa. Hence their mutual information is 0.
  * If X is a deterministic function of Y, then we can determine X from Y and Y from X with mutual information 1.
  * When we have Y = f(X,Z,M,N), 0 < mutual information < 1
  * We can select our features from feature space by ranking their mutual information with the target variable.
  * Advantage of using mutual information over F-Test is, it does well with the non-linear relationship between feature and target variable.
  * Sklearn offers feature selection with Mutual Information for regression and classification tasks.
  * sklearn.feature_selection.mututal_info_regression 
  * sklearn.feature_selection.mututal_info_classif
  * I(X ; Y) = H(X) – H(X | Y). Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X and H(X | Y) is the conditional entropy for X given Y. The result has the units of bits. Mutual information is a measure of dependence or “mutual dependence” between two random variables. As such, the measure is symmetrical, meaning that I(X ; Y) = I(Y ; X).
* PCA (Principal COmponent Analysis) - Linear combination of the variables
  * https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643
  * PCA extracts low dimensional set of features from a high dimensional data set such that variance is maximized/increased by bringing the data into low dimensional space. Loses a little accuracy. Normalization of data is performed before performing PCA.
* Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components. Because our principal components are orthogonal to one another, they are statistically independent of one another.
  * Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables. These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components.
  * First principal component is a linear combination of original predictor variables which captures the maximum variance in the data set. It determines the direction of highest variability in the data. 
  * Second principal component (Z²) is also a linear combination of original predictors which captures the remaining variance in the data set and is uncorrelated with Z¹.
  * PCA is a linear algorithm
* LDA - Linear Discriminant Analysis 
  * LDA approach is very similar to a Principal Component Analysis, but in addition to finding the component axes that maximize the variance of our data (PCA), we are additionally interested in the axes that maximize the separation between multiple classes (LDA).
  * PCA is unsupervised, LCA is supervised
  * Principal component analysis involves extracting linear composites of observed variables.
* t-Distributed Stochastic Neighbor Embedding (t-SNE)
  * https://distill.pub/2016/misread-tsne/
  * (t-SNE) t-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction algorithm used for exploring high-dimensional data. It maps multi-dimensional data to two or more dimensions suitable for human observation. 
  * t-SNE is based on probability distributions with random walk on neighborhood graphs to find the structure within the data.
  * A major problem with, linear dimensionality (PCA) reduction algorithms is that they concentrate on placing dissimilar data points far apart in a lower dimension representation. But in order to represent high dimension data on low dimension, non-linear manifold, it is important that similar datapoints must be represented close together, which is not what linear dimensionality reduction algorithms do.
  * t-SNE is capable of retaining both the local and global structure of the data at the same time. Local approaches seek to map nearby points on the manifold to nearby points in the low-dimensional representation. Global approaches on the other hand attempt to preserve geometry at all scales, i.e mapping nearby points to nearby points and far away points to far away points  
  * t-SNE a non-linear dimensionality reduction algorithm finds patterns in the data by identifying observed clusters based on similarity of data points with multiple features. But it is not a clustering algorithm it is a dimensionality reduction algorithm. This is because it maps the multi-dimensional data to a lower dimensional space, the input features are no longer identifiable. 
  * Steps of ALgo
    * https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
    * https://www.kdnuggets.com/2018/08/introduction-t-sne-python.html
* Use Algorithms that have built in feature reduction techniques
* Factor analyis - Variables are grouped by their correlations i.e. all variables in a particular group will have a high correlation among themselves.
  * Factor analysis is based on a formal model predicting observed variables from theoretical latent factors. https://www.theanalysisfactor.com/the-fundamental-difference-between-principal-component-analysis-and-factor-analysis/
* Dimensionality reduction: 
  * Multiple Correspondence Analysis (MCA) - Categorical variables
  * CorEx - Recent technique for automatic structure extraction from categorical data - https://github.com/gregversteeg/CorEx
  * Self-organizing maps (SOM)
  * Latent Semantic Indexing
* Variational autoencoders: An automated generative approach using variational autoencoders (VAE)
* Clustering: Hierarchical Clustering
* Measure information gain for the available set of features and select top n features accordingly
* Use regression and select variables based on p values






