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
* Inter Quartile Range - Beyond the range of -1.5 x IQR to 1.5 x IQR
* Out of range of 5th and 95th percentile
* Student Residuals / Z-score - 3 or more standard deviation away from mean
* Mahalanobis’ distance
* Cook’s Distance plot - Residuals vs Leverage Plot
* Cook’s D Bar Plot
* Cook’s D Chart
* DFBETAs Panel
* DFFITs Plot
* Studentized Residual Plot
* Standardized Residual Chart
* Studentized Residuals vs Leverage Plot
* Deleted Studentized Residual vs Fitted Values Plot
* Hadi Plot
* Potential Residual Plot

### Outlier Treatment
* Understand why it is happening then use business knowledge
* Delete obs
* Transforming and binning values
* Imputing
* Treat seperately
* Use algorithms robust to outliers - like Decision trees

### PyOD Library


