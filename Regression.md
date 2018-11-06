Regression is a method of modelling a target value based on independent predictors. Explains the degree of relationship between 2 or more variables using the bets fit line.

### Assumptions
##### Linearity
There should be a linear and additive relationship between dependent (response) variable and independent (predictor) variable(s). A linear relationship suggests that a change in response Y due to one unit change in X¹ is constant, regardless of the value of X¹. An additive relationship suggests that the effect of X¹ on Y is independent of other variables.
* Residual(Diff in actual & pred) vs Fitted values(Pred): Linearity assumption is satisfied if residuals are spread out randomly around the 0 line
##### Autocorrelation
There should be no correlation between the residual terms. Absence of correlation is known as Autocorrelation.
If the error terms are correlated, the estimated standard errors tend to underestimate the true standard error. If this happens, it causes confidence intervals and prediction intervals to be narrower. 
This usually occurs in time series models where the next instant is dependent in the previous instant.
* Durbin – Watson (DW) statistic: It must lie between 0 and 4. If DW = 2, implies no autocorrelation, 0 < DW < 2 implies positive autocorrelation while 2 < DW < 4 indicates negative autocorrelation.
* Residual vs time plot and look for the seasonal or correlated pattern in residual values.
##### Multicollinearity
The independent variables should not be correlated. Absence of this phenomenon is known as multicollinearity.
Standard errors tend to increase in presence of multicollinearity. With large standard errors, the confidence interval becomes wider leading to less precise estimates of slope parameters.
* VIF factor: VIF value <= 4 suggests no multicollinearity whereas a value of >= 10 implies serious multicollinearity. 

##### Heteroskedasticity
The error terms must have constant variance. This phenomenon is known as homoskedasticity. The presence of non-constant variance is referred to heteroskedasticity.
* Residual(Diff in actual & pred) vs Fitted values(Pred) Plot: 
##### Normal Distribution of error terms
* The error terms must be normally distributed.

### Ordinary Least Sqaure (OLS) Algorithm
It is used in python library sklearn. 

### Gradient Descent Algorithm
A gradient measures how much the output of a function changes if you change the inputs a little bit. <br/>
Gradient descent is an optimization algorithm that finds the optimal weights (a,b) (Equation: Y=a+bX) that reduces prediction error i.e difference between actual and predicted values. Steps:
1. Initialize the weights(a & b) with random values and calculate Error (SSE)
2. Calculate the gradient i.e. change in SSE when the weights (a & b) are changed by a very small value from their original randomly initialized value. This helps us move the values of a & b in the direction in which SSE is minimized.
3. Adjust the weights with the gradients to reach the optimal values where SSE is minimized
4. Use the new weights for prediction and to calculate the new SSE
5. Repeat steps 2 and 3 till further adjustments to weights doesn’t significantly reduce the Error

Learning Rate: Determines how fast or slow we will move towards the optimal weights. In order for Gradient Descent to reach the local minimum, we have to set the learning rate to an appropriate value, which is neither too low nor too high. This is because if the steps it takes are too big, it maybe will not reach the local minimum and if you set the learning rate to a very small value, gradient descent will eventually reach the local minimum but it will maybe take too much time.
##### Batch Gradient Descent
It is also called vanilla gradient descent, calculates the error for each example within the training dataset, but only after all training examples have been evaluated, the model gets updated. This whole process is like a cycle and called a training epoch. <br/>
Parameters are updated only after evaluating all examples within training set are evaluated. Takes big, slow steps. <br/> 
It produces a stable error gradient and a stable convergence. <br/>
It is preferable for small datasets. <br/>
##### Stochastic Gradient Descent
Parameters are updated for each example within the training set, one by one. It takes small, quick steps. <br/>
The frequent updates (computationally expensive) allow us to have a pretty detailed rate of improvement. <br/>
It is preferable for large datasets. <br/>
##### Mini Batch Gradient Descent
It simply splits the training dataset into small batches and performs an update on the parameters for each of these batches. <br/> 
Common mini-batch sizes range between 50 and 256. <br/>

### Evaluation Metrics
##### RMSE - Root Mean Squared Error
It is the sample standard deviation of the differences between predicted values and observed values (called residuals).  <br/>
RMSE = ( {Σ(Y – Ypred)²} / n )^0.5 <br/>
RMSE is higher or equal to MAE and is the default metric in most models because loss function defined in terms of RMSE is smoothly differentiable and makes it easier to perform mathematical operations. <br/>
Minimizing the squared error over a set of numbers results in finding its mean.
##### MAE - Mean Absolute error
MAE = (Σ|Y – Ypred|) / n  <br/>
Minimizing the absolute error results in finding its median. <br/> 
##### MSE - Mean Squared error
MSE = {Σ(Y – Ypred)²} / n 
##### R Squared (R²) / Coeffiecient of Determination
It is the proportion of variance in the response variable that is explained by the independant variables. It represents how close the data values are to the fitted regression line. <br/>
Ranges from 0 to 1 and are commonly stated as percentages from 0% to 100%. <br/>
R² = 1 - (Explained Variation by model / Total Variation) <br/>
Explained variation by model (It is not the complete formulae for variance) = Σ(Y – Ypred)² <br/> 
Total Variation (It is not the complete formulae for variance) = Σ(Y – Yavg)² <br/>
##### Adjusted R Squared (R²)
R² assumes that every single variable explains the variation in the dependent variable. R² either stay the same or increase with addition of more variables wven if they dont have any relationship with the output variables. <br/> 
The adjusted R² tells you the percentage of variation explained by only the independent variables that actually affect the dependent variable. It penalizes you for adding variables which dont improve your existing model. <br/> 
R-square and Adjusted R squared would be exactly same for single input variable. <br/> 
Adjusted R² = 1 - { (1-R²)(n-1)/(n-k-1) }   
n is total sample size and k is no of predictors




Sum of Squared Errors (SSE) = =  {Σ(Y – Ypred)²} / 2




MSE - Mean Squared error


