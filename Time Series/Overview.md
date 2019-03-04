### Links
ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/StatisticsMachineLearningPythonDraft.pdf <br/>
https://pandas.pydata.org/pandas-docs/stable/timeseries.html <br/>
http://www.statsmodels.org/devel/tsa.html <br/>
https://otexts.org/fpp2/index.html <br/>
https://robjhyndman.com/hyndsight/cyclicts/ <br/>
http://www.statsoft.com/Textbook/Time-Series-Analysis <br/>

### Overview
The three most widely used are regression models (Method of Least Squares), smoothing models. and general time series models. <br/>
Prediction Interval (PI) is LCL - UCL <br/>
Time series models are very useful models when you have serially correlated data. 
* Trend:
* Seasonal: A seasonal pattern exists when a series is influenced by seasonal factors (e.g., the quarter of the year, the month, or day of the week). Seasonality is always of a fixed and known period. Hence, seasonal time series are sometimes called periodic time series.
* Cyclic: A cyclic pattern exists when data exhibit rises and falls that are not of fixed period. The duration of these fluctuations is usually of at least 2 years. Think of business cycles which usually last several years, but where the length of the current cycle is unknown beforehand.

### Stationarity
Unless your time series is stationary, you cannot build a time series model. <br/>
Conditions for a series to be classified as stationary series:
* The mean of the series should not be a function of time rather should be a constant. 
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Mean_nonstationary.png)
* The variance of the series should not a be a function of time.
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Var_nonstationary.png)
* The covariance of the i th term and the (i + m) th term should not be a function of time. An autocovariance that does not depend on time. Autocovariance is a function that gives the covariance of the process with itself at pairs of time points.
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Cov_nonstationary.png)

##### Reasons for Stationarity
* Trend – varying mean over time. For eg, in this case we saw that on average, the number of passengers was
growing over time.
* Seasonality – variations at specific time-frames. eg people might have a tendency to buy cars in a particular
month because of pay increment or festivals.

White noise is stationary - it does not matter when you observe it, it should look much the same at any point in time.
Ways to Stationarize Non-Stationary models:
* Detrending
* Differencing - Try double differencing too
* Transformation - Used only in case differencing is not working.
  * Log - Incase of diverging time series
  

Cross validation for time series - https://www.r-bloggers.com/cross-validation-for-time-series/  evaluation on a rolling forecasting origin - onestep or multistep forecast (Training data - A points, Test data - k points) For cross validation every time one observation from the test data is added to training data to determine the immediate next value) - Average of all the test points is used to finally determine the model accuracy


