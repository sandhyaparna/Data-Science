Regression is a method of modelling a target value based on independent predictors. Explains the degree of relationship between 2 or more variables using the best fit line.

### Assumptions
##### Linearity
There should be a linear and additive relationship between dependent (response) variable and independent (predictor) variable(s). A linear relationship suggests that a change in response Y due to one unit change in X¹ is constant, regardless of the value of X¹. An additive relationship suggests that the effect of X¹ on Y is independent of other variables. <br/>
* Residual(Diff in actual & pred) vs Fitted values(Pred): Linearity assumption is satisfied if residuals are spread out randomly around the 0 line <br/>
To overcome the issue of non-linearity, you can do a non linear transformation of predictors such as log (X), √X or X² transform the dependent variable.
##### No Autocorrelation
There should be no correlation between the residual terms. Absence of this phenomenon is known as Autocorrelation. <br/>
If the error terms are correlated, the estimated standard errors tend to underestimate the true standard error. If this happens, it causes confidence intervals and prediction intervals to be narrower.  <br/>
This usually occurs in time series models where the next instant is dependent in the previous instant. <br/>
* Durbin – Watson (DW) statistic: It must lie between 0 and 4. If DW = 2, implies no autocorrelation, 0 < DW < 2 implies positive autocorrelation while 2 < DW < 4 indicates negative autocorrelation. <br/>
* Residual vs time plot and look for the seasonal or correlated pattern in residual values. <br/>
##### No Multicollinearity
The independent variables should not be correlated. Absence of this phenomenon is known as multicollinearity. <br/>
Standard errors tend to increase in presence of multicollinearity. With large standard errors, the confidence interval becomes wider leading to less precise estimates of slope parameters. <br/>
* VIF factor: VIF value <= 4 suggests no multicollinearity whereas a value of >= 10 implies serious multicollinearity.  <br/>
##### Homoskedasticity
The error terms must have constant variance. This phenomenon is known as homoskedasticity. The presence of non-constant variance is referred to heteroskedasticity. <br/>
Heteroskedasticity i.e non-constant variance often creates cone-line shape scatter plot of residuals vd fitted. Scattering widens or narrows as the value of fitted increases implies it is consistently accurate when it predicts low values, but highly inconsistent in accuracy when it predicts high values. <br/>
* Residual(Diff in actual & pred) vs Fitted values(Pred) Plot: Should not be cone-shaped/funnel shaped <br/>
* Scale Location Plot ( Square root of Standardized residuals vs Theoritical Quantiles) <br/>
To overcome heteroskedasticity, a possible way is to transform the response variable such as log(Y) or √Y. Also, you can use weighted least square method to tackle heteroskedasticity. <br/>
##### Normal Distribution of error terms
The error terms must be normally distributed. If not, confidence intervals may become too wide or narrow. Once confidence interval becomes unstable, it leads to difficulty in estimating coefficients based on minimization of least squares. <br/>
* Q-Q plot (Standardized residuals vs Theoritical Quantiles) <br/>
If the errors are not normally distributed, non–linear transformation of the variables (response or predictors) can bring improvement in the model.
###### No outliers
Linear Regression is very sensitive to Outliers. It can terribly affect the regression line and eventually the forecasted values.

### Ordinary Least Sqaure (OLS) Algorithm
OLS minimizes residual sum of squares <br/>
It is used in python library sklearn.  <br/>

### Gradient Descent Algorithm
Main objective is to minimize cost function
A gradient measures how much the output of a function changes if you change the inputs a little bit. <br/>
Gradient descent is an optimization algorithm that finds the optimal weights (a,b) (Equation: Y=a+bX) that reduces prediction error i.e difference between actual and predicted values. Steps:
1. Initialize the weights(a & b) with random values and calculate Error (SSE)
2. Calculate the gradient i.e. change in SSE when the weights (a & b) are changed by a very small value from their original randomly initialized value. This helps us move the values of a & b in the direction in which SSE is minimized.
3. Adjust the weights with the gradients to reach the optimal values where SSE is minimized
4. Use the new weights for prediction and to calculate the new SSE
5. Repeat steps 2 and 3 till further adjustments to weights doesn’t significantly reduce the Error <br/>
Learning Rate: Determines how fast or slow we will move towards the optimal weights. In order for Gradient Descent to reach the local minimum, we have to set the learning rate to an appropriate value, which is neither too low nor too high. This is because if the steps it takes are too big, it maybe will not reach the local minimum and if you set the learning rate to a very small value, gradient descent will eventually reach the local minimum but it will maybe take too much time. <br/>
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

### Dimensionality Reduction
##### Forward Selection
Starts with most significant predictor in the model and adds variable for each step.
##### Backward Elimination
Starts with all predictors in the model and removes the least significant variable for each step.
##### Stepwise Selection
It does two things. It adds and removes predictors as needed for each step.

### Regularization
Regularization basically adds the penalty as model complexity increases. Regularization is used to prevent the model from overfitting the training sample. It constrains/regularizes or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. In regularization, we normally keep the same number of features, but reduce the magnitude of the coefficients.
#####  Ridge Regression - L2 Regularization
Ridge Regression is a technique used when the data suffers from multicollinearity ( independent variables are highly correlated). It solves the multicollinearity problem through shrinkage parameter λ (lambda), shrinks the value of coefficients but doesn’t reaches zero. <br/>
Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function. <br/>
Minimization objective = Least Squares Obj + α * (sum of square of coefficients) <br/>
Magnitude of coefficients decreases as λ increases. λ basically controls penality term in the cost function of ridge reg. <br/>
R² for a range of λ and choose the one that gives higher R². <br/>
If the penality is very large it means model is less complex, therefore the bias would be high. <br/>
* Assumptions of this regression is same as least squared regression except normality is not to be assumed. <br/>
#####  Lasso Regression - L1 Regularization
Coefficients are reducing to 0 even for smaller changes in λ. Lasso selects the only some feature while reduces the coefficients of others to zero. This property is known as feature selection and which is absent in case of ridge. <br/>
Least Absolute Shrinkage and Selection Operator (Lasso) adds “absolute value of magnitude” of coefficient as penalty term to the loss function. <br/>
Minimization objective = Least Squares Obj + α * (sum of absolute value of coefficients) <br/>
If group of predictors are highly correlated, lasso picks only one of them and shrinks the others to zero. <br/>
* Assumptions of this regression is same as least squared regression except normality is not to be assumed. <br/>
##### Elastic Net Regression
It generally works well when we have a big dataset. <br/>
Elastic net is basically a combination of both L1 and L2 regularization. <br/>
Elastic regression working: Let’ say, we have a bunch of correlated independent variables in a dataset, then elastic net will simply form a group consisting of these correlated variables. Now if any one of the variable of this group is a strong predictor (meaning having a strong relationship with dependent variable), then we will include the entire group in the model building, because omitting other variables (like what we did in lasso) might result in losing some information in terms of interpretation ability, leading to a poor model performance. <br/>

### Evaluation Metrics
Sum of Squared Errors (SSE) = =  {Σ(Y – Ypred)²} / 2
##### RMSE - Root Mean Squared Error
It is the sample standard deviation of the differences between predicted values and observed values (called residuals).  <br/>
RMSE = ( {Σ(Y – Ypred)²} / n )^0.5 <br/>
RMSE is higher or equal to MAE and is the default metric in most models because loss function defined in terms of RMSE is smoothly differentiable and makes it easier to perform mathematical operations. <br/>
Minimizing the squared error over a set of numbers results in finding its mean. <br/>
##### MAE - Mean Absolute error
MAE = (Σ|Y – Ypred|) / n  <br/>
Minimizing the absolute error results in finding its median. <br/> 
##### MSE - Mean Squared error
MSE = {Σ(Y – Ypred)²} / n  <br/>
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
Adjusted R² = 1 - { (1-R²)(n-1)/(n-k-1) }    <br/>
n is total sample size and k is no of predictors <br/>

### Bullet Points
* Least Square Error is used to identify the line of best fit
* Overfitting is more likeley when we 
* Sum of residuals is always 0 in linear reg
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


