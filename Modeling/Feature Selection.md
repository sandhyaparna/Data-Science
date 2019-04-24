### Links
Genetic Algorithm https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29 <br/>
https://topepo.github.io/caret/feature-selection-using-genetic-algorithms.html <br/>
Dimension reduction techniques https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/ <br/>
Filter, Wrapper Methods https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/ <br/>

### Reasons to include fewer predictors over many
* Redundancy/Irrelevance: Remove non-redundant predictor variables
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
* LDA - Linear Discriminant Analysis 
  * LDA approach is very similar to a Principal Component Analysis, but in addition to finding the component axes that maximize the variance of our data (PCA), we are additionally interested in the axes that maximize the separation between multiple classes (LDA).
  * PCA is unsupervised, LCA is supervised
  * Principal component analysis involves extracting linear composites of observed variables.
* Factor analyis - Variables are grouped by their correlations i.e. all variables in a particular group will have a high correlation among themselves.
  * Factor analysis is based on a formal model predicting observed variables from theoretical latent factors. https://www.theanalysisfactor.com/the-fundamental-difference-between-principal-component-analysis-and-factor-analysis/
* Use Algorithms that have built in feature reduction techniques
* Dimensionality reduction: 
  * Principal Component Analysis (PCA) - Linear combination of the original variables
  * Multiple Correspondence Analysis (MCA) - Categorical variables
  * CorEx - Recent technique for automatic structure extraction from categorical data - https://github.com/gregversteeg/CorEx
  * Self-organizing maps (SOM)
  * Latent Semantic Indexing
* Manifold Learning: t-Distributed Stochastic Neighbor Embedding (t-SNE)
* Variational autoencoders: An automated generative approach using variational autoencoders (VAE)
* Clustering: Hierarchical Clustering
* Measure information gain for the available set of features and select top n features accordingly
* Use regression and select variables based on p values






