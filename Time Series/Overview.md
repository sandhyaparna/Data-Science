### Links
ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/StatisticsMachineLearningPythonDraft.pdf <br/>
https://pandas.pydata.org/pandas-docs/stable/timeseries.html <br/>
http://www.statsmodels.org/devel/tsa.html <br/>


### Overview
Time series models are very useful models when you have serially correlated data.


### Stationarity
Unless your time series is stationary, you cannot build a time series model. <br/>
Conditions for a series to be classified as stationary series:
* The mean of the series should not be a function of time rather should be a constant. 
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Mean_nonstationary.png)
* The variance of the series should not a be a function of time.
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Var_nonstationary.png)
* The covariance of the i th term and the (i + m) th term should not be a function of time.
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Cov_nonstationary.png)

Ways to Stationarize Non-Stationary models:
* Detrending
* Differencing


