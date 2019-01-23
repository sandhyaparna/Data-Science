### Links
https://www.saedsayad.com/logistic_regression.htm


### Overview
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
MLE determines the regression coefficient that accurately predicts the probability of the binary dependent variable <br/>


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

