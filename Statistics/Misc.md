### Outliers
Outlier is an observation that appears far away and diverges from an overall pattern in a sample
* Univariate Outliers: Distribution of a single variable - Box-plot, Histogram
* Multivariate Outliers: Distributions in multi-dimensions (Eg- For a particular height, certain weight may be an outlier) - Scatter Plot
##### Detecting Outliers
* Beyond the range of -1.5 x IQR to 1.5 x IQR
* Out of range of 5th and 95th percentile
* 3 or more standard deviation away from mean
Cook’s Distance plot - Residuals vs Leverage Plot


### Simpson's paradox
Simpson's paradox occurs when groups of data show one particular trend, but this trend is reversed when the groups are combined together. Understanding and identifying this paradox is important for correctly interpreting data.

Simpson’s paradox refers to a phenomena whereby the association between a pair of variables (X, Y ) reverses sign upon conditioning of a third variable, Z, regardless of the value taken by Z. If we partition the data into subpopulations, each representing a specific value of the third variable, the phenomena appears as a sign reversal between the associations measured in the disaggregated subpopulations relative to the aggregated data, which describes the population as a whole.

Check out the example here - https://www.thoughtco.com/what-is-simpsons-paradox-3126365

### Biased Estimate
When the expected mean of the sampling distribution of a statistic is not equal to a population parameter, that statistic is said to be a biased estimate of the parameter

### Bessels Correction
Bessels Correction while calculating a sample standard deviation is used when we are trying to estimate population standard deviation from the sample, it is less biased

