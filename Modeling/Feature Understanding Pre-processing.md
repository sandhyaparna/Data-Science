https://www.kdnuggets.com/2018/11/secret-sauce-top-kaggle-competition.html <br/>

* Features wrt Target
* Identify noisy features - Noisy features lead to overfitting. Features are noisy if trends in train and test dont match. 
  * Trend Correlation - If a feature doesn’t hold same trend w.r.t. target across train and evaluation sets, it can lead to overfitting. 
  * Trend changes - Sudden and repeated changes in trend direction could imply noisiness. But, such trend change can also happen because that bin has a very different population in terms of other features and hence, its default rate can’t really be compared with other bins.
* Missing data analysis
* Number of Unique values within Categorical data

### Data-Preprocessing
* Remove examples that you dont want to train on
* COmpute vocabularies for cat columns
* Compute aggregate stats for numeric cols
* Compute time-windowed stats like last hr, last month etc
* 





