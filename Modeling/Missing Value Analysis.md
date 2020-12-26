### Links
https://medium.freecodecamp.org/the-penalty-of-missing-values-in-data-science-91b756f95a32 </br>

### Handling
* Remove Variables - Maximum limit for number of missing values = 25-30%
* Remove Rows - Deletion
* Generalized Imputation - Mean/Median/Mode of non-missing values
* Similar case Imputation - Group wise imputation 
* Reconstruction - KNN imputation / using predictive model 

### Imputation
* XGBoost can handle missing values
* Median - Continuous data
* Mode - Categorical data
* Mode, Median in combination with groupby
* Forward or Backward fill
* Replace missing with 0
* Soft Probabilities - Replace NaNs randomly in a ratio which is “proportional” to the population without NaNs (the proportion is calculated using probabilities but with a touch of randomness)
   * Better data distribution
   * Less Biased
   * Successful conservation of mean
   * Chnace of over-fitting a model using this data is less compared to hard imputing with mean, median or mode
   * Calculate Probability & Expected value -https://www.freecodecamp.org/news/the-penalty-of-missing-values-in-data-science-91b756f95a32/
* KNN Imputation: https://www.youtube.com/watch?v=AHBHMQyD75U </br>
Missing values in independent variables are taken into account by using weightage. For eg: if the row we are using for comparing to calculate distance have missing values in 2 independent vars, weightage = Num of non-missing vars/Total no of vars 
  * Disadvantages : time consuming on large datasets; on high dimensional data, accuracy can be severely degraded
* MICE (Multivariate Imputation by Chained Equation). It works on the assumption that missing data are Missing at Random (MAR) </br>
https://medium.com/swlh/mice-algorithm-to-impute-missing-values-in-a-dataset-c55d555b6fbe </br>
  * Disadvantages : No theoretical justifications as other imputation methods; Data complexities
* MissForest: imputation algorithm that operates on the Random Forest algorithm </br>
https://towardsdatascience.com/missforest-the-best-missing-data-imputation-algorithm-4d01182aed3 </br>
  * 
* DataWig learns Machine Learning models to impute missing values in tables.
* Feature Hashing - The logic by which a hash is calculated depends on the hash function itself, but all hash functions share the same common characteristics:
  * If we feed the same input to a hash function, it will always give the same output.
  * The choice of hash function determines the range of possible outputs, i.e. the range is always fixed (e.g. numbers from 0 to 1024).
  * Hash functions are one-way: given a hash, we can’t perform a reverse lookup to determine what the input was.
  * Hash functions may output the same value for different inputs (collision).




### Techniques
* Mean is not used
* Do Nothing or create a new category for missing in Categorical data - Is_missing_or_not column and then impute the original var with the missing values
* Median, Mode are usually used in linear models
* Missing values should be imputed befor Feature engineering
* Missing values needs to be ignored in the imputation process(calculating median etc)
* For categories which present in the test data but do not present in the train data - frequency encoding works well


