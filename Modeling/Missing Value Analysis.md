### Links
https://medium.freecodecamp.org/the-penalty-of-missing-values-in-data-science-91b756f95a32 </br>

### Handling
* Do nothing and use ML algos like XGBoost, LightGBM (LightGBM — use_missing=false) that can handle missing values
* Remove Variables - Maximum limit for number of missing values = 25-30%
* Remove Rows - Deletion/Listwise deletion - Introduces a lot of bias
* Generalized Imputation - Mean/Median/Mode of non-missing values
  * Easy, fast and works well with small numerical datasets
  * k to use if missing data is less than 3%, otherwise introduces too much bias and artificially lowers variability of data
  * Doesn’t factor the correlations between features. It only works on the column level
  * Doesn’t account for the uncertainty in the imputations
  * Not very accurate
  * It can introduce bias in the data
* Similar case Imputation - Group wise imputation 
* previous or next values
* Reconstruction - KNN imputation / using predictive model / MICE

Missing data mechanism describes the underlying mechanism that generates missing data and can be categorized into three types — missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR).

### Imputation
* XGBoost can handle missing values
* Median - Continuous data
* Mode - Categorical data
* Mode, Median in combination with groupby
* Forward or Backward fill
* Replace missing with 0
data-science-91b756f95a32/
* KNN Imputation: https://www.youtube.com/watch?v=AHBHMQyD75U </br>
Missing values in independent variables are taken into account by using weightage. For eg: if the row we are using for comparing to calculate distance have missing values in 2 independent vars, weightage = Num of non-missing vars/Total no of vars 
  * Disadvantages : time consuming on large datasets; on high dimensional data, accuracy can be severely degraded
* MICE (Multivariate Imputation by Chained Equation). It works on the assumption that missing data are Missing at Random (MAR) </br>
https://medium.com/swlh/mice-algorithm-to-impute-missing-values-in-a-dataset-c55d555b6fbe </br>
  * Disadvantages : No theoretical justifications as other imputation methods; Data complexities
* MissForest: imputation algorithm that operates on the Random Forest algorithm </br>
https://towardsdatascience.com/missforest-the-best-missing-data-imputation-algorithm-4d01182aed3 </br>
* Hot Deck Imputation (couldn't find python code but it is similar to bfill/ffill) : Find all the sample subjects who are similar on other variables, then randomly choose one of their values to fill in. Good because constrained by pre-existing values, but the randomness introduces hidden variability and is computationally expensive
* Cold Deck Imputation (couldn't find python code but it is similar to bfill/ffill): Systematically choose the value from an individual who has similar values on other variables (e.g. the third item of each collection). This option removes randomness of hot deck imputation. Positively constrained by pre-existing values, but the randomness introduces hidden variability and is computationally expensive

##### Not frequently used
* Soft Probabilities - Replace NaNs randomly in a ratio which is “proportional” to the population without NaNs (the proportion is calculated using probabilities but with a touch of randomness)
   * Better data distribution
   * Less Biased
   * Successful conservation of mean
   * Chnace of over-fitting a model using this data is less compared to hard imputing with mean, median or mode
   * Calculate Probability & Expected value -https://www.freecodecamp.org/news/the-penalty-of-missing-values-in-
* DataWig learns Machine Learning models to impute missing values in tables.
  * Single Column imputation
  * Can be quite slow with large datasets
  * You have to specify the columns that contain information about the target column that will be imputed
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


