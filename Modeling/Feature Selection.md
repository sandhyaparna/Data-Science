### Links
Genetic Algorithm https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29 <br/>
https://topepo.github.io/caret/feature-selection-using-genetic-algorithms.html <br/>
Dimension reduction techniques https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/ <br/>
Filter, Wrapper Methods https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/ <br/>

### Reasons to include fewer predictors over many
* Redundancy/Irrelevance: Remove non-redundant predictor variables. Pareto principle suggests that 80% of what happens can be explained using 20%
* Garbage In = Garbage Out - So, we are trying to avoid it
* Over-fitting: The data models with large number of predictors (also referred to as complex models) often suffer from the problem of overfitting, in which case the data model performs great on training data, but performs poorly on test data.
* Productivity: Improves Accuracy (Less Noise), Reduces Training time
* Understandability: Fewer predictors are way easier to understand and explain

### Feature reduction techniques: Numeric & Categorical - Domain dependent
* Missing value ratio
* Low Variance filter
* Remove highly correlated variables
* Regularization - Lasso
* Forward slection, Backward Selection, Stepwise Selection
* Chi-sq for Categorical data - SelectKBest based on chi-sq score in python 
* PCA (Principal COmponent Analysis) - Linear combination of the variables
  * PCA extracts low dimensional set of features from a high dimensional data set such that variance is maximized/increased by bringing the data into low dimensional space. Loses a little accuracy. Normalization of data is performed before performing PCA.
* Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components. Because our principal components are orthogonal to one another, they are statistically independent of one another.
  * Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables. These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components.
  * First principal component is a linear combination of original predictor variables which captures the maximum variance in the data set. It determines the direction of highest variability in the data. 
  Second principal component (Z²) is also a linear combination of original predictors which captures the remaining variance in the data set and is uncorrelated with Z¹.
* LDA - Linear Discriminant Analysis 
  * LDA approach is very similar to a Principal Component Analysis, but in addition to finding the component axes that maximize the variance of our data (PCA), we are additionally interested in the axes that maximize the separation between multiple classes (LDA).
  * PCA is unsupervised, LCA is supervised
  * Principal component analysis involves extracting linear composites of observed variables.
* Factor analyis - Variables are grouped by their correlations i.e. all variables in a particular group will have a high correlation among themselves.
  * Factor analysis is based on a formal model predicting observed variables from theoretical latent factors. https://www.theanalysisfactor.com/the-fundamental-difference-between-principal-component-analysis-and-factor-analysis/
* Use Algorithms that have built in feature reduction techniques
* Dimensionality reduction: 
  * Multiple Correspondence Analysis (MCA) - Categorical variables
  * CorEx - Recent technique for automatic structure extraction from categorical data - https://github.com/gregversteeg/CorEx
  * Self-organizing maps (SOM)
  * Latent Semantic Indexing
* Manifold Learning: t-Distributed Stochastic Neighbor Embedding (t-SNE)
* Variational autoencoders: An automated generative approach using variational autoencoders (VAE)
* Clustering: Hierarchical Clustering
* Measure information gain for the available set of features and select top n features accordingly
* Use regression and select variables based on p values






