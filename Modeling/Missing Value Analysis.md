### Links
https://medium.freecodecamp.org/the-penalty-of-missing-values-in-data-science-91b756f95a32 </br>

### Handling
* Remove Variables - Maximum limit for number of missing values = 25-30%
* Remove Rows - Deletion
* Generalized Imputation - Mean/Median/Mode of non-missing values
* Similar case Imputation - Group wise imputation 
* KNN imputation / using predictive model

### Imputation
* Median - Continuous data
* Mode - Categorical data
* Forward or Backward fill
* Replace missing with 0
* Soft Probabilities - Replace NaNs randomly in a ratio which is “proportional” to the population without NaNs (the proportion is calculated using probabilities but with a touch of randomness)
   * Better data distribution
   * Less Biased
   * Successful conservation of mean
   * Chnace of over-fitting a model using this data is less compared to hard imputing with mean, median or mode
   * Calculate Probability & Expected value -https://www.freecodecamp.org/news/the-penalty-of-missing-values-in-data-science-91b756f95a32/
* MICE (Multivariate Imputation by Chained Equation)
* DataWig learns Machine Learning models to impute missing values in tables.

### Techniques
* Mean is not used
* Do Nothing or create a new category for missing in Categorical data
* 





