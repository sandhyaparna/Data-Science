https://github.com/hse-aml/competitive-data-science
* Data Pre-processing
  * Numeric - Transformation
  * categorical - Encoding, create interaction vars by concatenating vars 
  * Missing - Replace with a diff value, mean, median; innull_feature, 
* EDA
  * Domain knowledge, Understanding how data was generated, check if data agrees with our domain knowledge
  * Anonymized data - Guess meaning of columns, types of column. Explore feature relations - relations between pairs or find feature groups
  * Visualization - Histograms, plots(index vs values) - plt.plot(x,'.'), stats. Scatter plots - pd.scatter_matrix(df), Correlation (sorted) plots - df.corr(). Groups - plt.matshow(..); df.mean().plot(style='.'); df.mean().sort_values().plot(style='.')
    * If there is a peak at mean value - it can imply that organizers imputed missing value with mean
    * color coded can be based on 
  * Cleaning - Constant features, Duplicated features - cat features can also be identical but their levels have diff names , Duplicated rows 
    * Duplicated cat feat - for f in cat_features: Df[f] = raintest[f].factorize()
    * Duplicate rows - 
  * Check if dataset is shuffled - 
* Validation
* Data-Leakages



# ML Algos
* Linear - Logistic Regression, SVM
  * Good for Sparse High Dimensional data
* Tree Based - Decision Teee, Random Forest, Boosting - Divide & Conquer approach
* KNN - 
* Neural Nets - Frameworks used are Tensorflow, Keras, mxnet, PyTorch, Lasagne

* Data Pre-processing
* Outliers


###

### Feature Generation - Prior Knowledge & EDA






