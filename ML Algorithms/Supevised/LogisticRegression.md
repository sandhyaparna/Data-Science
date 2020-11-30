### Links
https://www.saedsayad.com/logistic_regression.htm <br/>
https://www.kdnuggets.com/2019/01/logistic-regression-concise-technical-overview.html <br/>

In logistic regression, linear combination of inputs are mapped to log odds. It predicts the probability of occurrence of an event by fitting data to a logic function.  <br/>
logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk <br/>

### Interpretation of Categorical and Continuous vars of a Logistic Reg
Odds Ratio = p/(1-p)
* For each predictor u get p-value and OddsRatio=exp(Xi). Xi is coeff. Odds ratio gre than 1 implies positive relation
* Categorical Var: Group(1) has OddsRatio times greater odds of Target happening, holding all other vars constant https://www.theanalysisfactor.com/odds-ratio-categorical-predictor/
* Continuous Var: A unit inc of a predictor inc the dependent var by log odds. It will inc the odds of Target happening by a factor of OddsRatio of this var, holding all other vars constant

### Effect on coefs, p-value, confidence Interval when there are correlated vars in the model
https://newonlinecourses.science.psu.edu/stat501/node/346/  <br/>
* When predictor variables are correlated, the estimated regression coefficient of any one variable depends on which other predictor variables are included in the model.
* When predictor variables are correlated, the precision of the estimated regression coefficients decreases as more predictor variables are added to the model.
* When predictor variables are correlated, the marginal contribution of any one predictor variable in reducing the error sum of squares varies depending on which other variables are already in the model.
* When predictor variables are correlated, hypothesis tests for βk = 0 may yield different conclusions depending on which predictor variables are in the model.
* High multicollinearity among predictor variables does not prevent good, precise predictions of the response within the scope of the model. 

### Overview 
Logistic regression is used to create a decision boundary to maximize the log likelihood of the classification probabilities.  <br/> 
In the case of a linear decision boundary, logistic regression wants to have each point and the associated class as far from the hydroplane as possible and provides a probability which can be interpreted as prediction confidence. <br/> 
Tries to minimize cross-entropy. <br/> 
 
Loss Function is Log loss(Binary label) or Cross Entropy loss(Multi-class label) - Goal is to minimize this value. <br/> 
Measures the performance of a classification model where the prediction input is a probability value between 0 and 1. <br/>
Log Loss takes into account the uncertainty of your prediction based on how much it varies from the actual label. It penalizes wrong predictions very strongly. <br/>
Log Loss = (-1/N) * [ Σ{ylog(p) + (1-y)log(1-p)} ] <br/>
Model parameters are updated with respect to the derivative of the loss function. Derivative or slope of Loss function provides direction and stepsize in our serach.
* If Loss function derivative is +ve then move towards left implies decrease weights and if the value is big then take a big stepsize
* If Loss function derivative is -ve then move towards right implies increase weights and if the value is small then take a small stepsize
It is important to add regularization to logistic regression because:
* Helps stop weights being driven to +/- infinity
* Helps logits stay away from asymptotes which can halt training
Logistic regression predicts the probability of occurrence of an event by fitting data to a logit function. Log odds of the outcome is modeled as a linear combination of the predictor variables. Logistic regression is a part of GLM that assumes a linear relationship between link function and independent variables in logit model. Estimates probabilities using underlying logistic function.
* Probability(For +ve outcome) ranges from 0 to 1 <br/> 
* odds = p/(1-p) = probability of event occurrence / probability of not event occurrence  <br/>
  * Odds is defined as the ratio of the chance of the event happening to that of non-happening of the event <br/>
  * Odds range from 0 to ∞ <br/>
  * ln(odds) = ln(p/(1-p)) <br/>
* logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk <br/>
    * Log odds range from - ∞ to +∞. Log odds is used to extend the range of the output as input vars may be continuous vars  <br/>
    * Slopes defines the steepness of the curve
    * Constant term move the curve left and right 
* Inverse of  ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk  is known as Sigmoid function and it gives an S-shaped curve, that has a value of probability ranging from 0<p<1.
* Any monotonic function tha maps the unit interval to the real number can be considered as link. GLM is used for analyzing linear and non-linear effects of continuous and categorical predictors on discrete or continuous response vars (uses link fucntion)
* Should be used when data is linearly seperable. This dividing plane is called a linear discriminant, because its linear in terms of its function and it helps the model discriminate between points belonging to different classes

### Maximum Likelihood Estimation
https://www.analyticsvidhya.com/blog/2018/07/introductory-guide-maximum-likelihood-estimation-case-study-r/ </br>
* MLE is used to find the coefficients (estimators) that accurately predicts the probability of the binary dependent variable, that minimizes the likelihood function
* MLE can be defined as a method for estimating population parameters (such as the mean and variance for Normal, rate (lambda) for Poisson, etc.) from sample data such that the probability (likelihood) of obtaining the observed data is maximized.
* To find MLE, we need to differentiate the log-likelihood function and set it to 0
  * First  differential of log-likelihood function is set to 0 - Score equation
  * Confidence in the MLE is quantified by the pointedness of the log-likelihood. 2nd differential is called Observed Information

### Assumptions
* Observations are independent of each other
* No multicollinearity among independent vars
* Assumes a linear relationship between link function and independent variables 
* Dependent Variable is not normally distributed
* Errors need to be independent but not normally distributed - No Normal distribution of error terms
* Homoscedasticity is not required
* Dependent variable in logistic regression is not measured on an interval or ratio scale

### Advantages
* It's fast, highly interpretable, doesn't require input features to be scaled, doesn't require any tuning, easy to regularize, and outputs are well-calibrated predicted probabilities
* Dependent or Independent variables are not normally distributed

### Disadvantages
* Cannot solve non-linear problems
* It is vulnerable to over-fitting
* Atleast 50 data points per predictor is necessary to achiev stable results

### Evaluation
* Wald test is used to test the statistical significance of coefficients in the model 

