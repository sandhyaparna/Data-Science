https://github.com/hse-aml/competitive-data-science


### Problem Statement
* Understanding Problem, Problem formalization
* Scope of the model - Model predicts 6hrs before onset of a disease, Before making the decision of discharge at a hospital
* What is time period that is looked at while creating observations for Target & Non-Target population

### Data Cleaning
* Cohort of interest
* Identify & resolves Discrepancies in data

### Data-preprocessing
* Univariate analysis - Data cleaning. Look for bad data (Numeric value in categorical var), Duplicates (Visit ID might not be unique when we combine data from diff clients); Duplicates(In online data, a user might have submitted entries multiple times); Typos in Categorical data (observe Freq table)
* Outlier treatment (Temp_F might have values less than 35 - one assumption can be that the temp value is actually Celsius and convert it into Fahrenheit)
* Missing value treatment
* Data Transformation - Right skewed/Left skewed
* Feature engineering - Numeric/Categorical/DateTime Vars - Transactional data/Temporal data
* Feature selection
* Split data into - Train, Validation & Test

### Model Building
* Overfitting - Cross Validation, Regularization
* Model Interpretation - LIME, shap

### Evaluation & Comparision - Regression, classification, Unsupervised
* Precision-Recall Curves, AUC Curves - To determine the cut-off for an effective model

# Competition Steps
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
* Model Buliding
  * Validation - Holdout, KFold, Leave One out - Stratified
* Model optimization - Hyper parameter tuning
* Data-Leakages

#### Pre-processing
* Tree Based models dont depend on scaling
* Non-Tree-Based - Numeric - hugely depend on scaling
  * Raising to power<1 - sqrt
  * log transformation
  * Scaling - MinMiaxScaler, StandardScaler
  * Rank Transformation - Handles outliers - sets spaces between sorted values to be equal
* When data has outliers - In non-tree models use - rank, llog, sqrt, winsorization






