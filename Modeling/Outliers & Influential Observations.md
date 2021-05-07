### Links
Detecting Influential obs https://cran.r-project.org/web/packages/olsrr/vignettes/influence_measures.html <br/>
PyOD https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/

### Overview
Outlier: An outlier is a data point that diverges from an overall pattern in a sample. An outlier has a large residual (the distance between the predicted value () and the observed value (y)). Outliers lower the significance of the fit of a statistical model because they do not coincide with the model's prediction.  <br/>

Influential Obs: An influential point is any point that has a large effect on the slope of a regression line fitting the data. They are generally extreme values. The process to identify an influential point begins by removing the suspected influential point from the data set. If this removal significantly changes the slope of the regression line, then the point is considered an influential point. <br/>

* An outlier is assumed being unwanted, unexpected, or a must-be-wrong value to the human’s knowledge but not a rare event which is possible but rare.
* Tree-based models, non-parametric tests are resistant to outliers


### Identify - More in the link
* Box plot / Histograms
* Inter Quartile Range - Beyond the range of -1.5 x IQR to 1.5 x IQR. IQR is the middle 50% of values when ordered from lowest to highest. IQR = Q3 - Q1
* Out of range of 5th and 95th percentile
* Student Residuals / Z-score - 3 or more standard deviation away from mean
* Mahalanobis’ distance
* Cook’s Distance plot - Residuals vs Leverage Plot
* Cook’s D Bar Plot
  * delete observations one at a time.
  * refit the regression model on remaining (n−1) observations
  * examine how much all of the fitted values change when the ith observation is deleted.
* Cook’s D Chart
* DFBETAs Panel
* DFFITs Plot
* Studentized Residual Plot
* Standardized Residual Chart
* Studentized Residuals vs Leverage Plot
* Deleted Studentized Residual vs Fitted Values Plot
* Hadi Plot
* Potential Residual Plot

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
  
### Outlier Treatment
* Understand why it is happening then use business knowledge
* Delete obs - if they are errors
* Transforming and binning values
* Imputing
* Treat seperately
* Use algorithms robust to outliers - like Decision trees
* Build 2 models - 1 for normal values and other for extreme values

### PyOD Library


