### Overview
Logistic regression predicts the probability of occurrence of an event by fitting data to a logit function. Log odds of the outcome is modeled as a linear combination of the predictor variables. Logistic regression is a part of GLM that assumes a linear relationship between link function and independent variables in logit model.
* Probability ranges from 0 to 1 <br/>
* odds = p/(1-p) = probability of event occurrence / probability of not event occurrence  <br/>
  * Odds is defined as the ratio of the chance of the event happening to that of non-happening of the event <br/>
  * Odds range from 0 to ∞ <br/>
  * ln(odds) = ln(p/(1-p)) <br/>
* logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk <br/>
    * Log odds range from - ∞ to +∞. Log odds is used to extens the range of the output as input vars may be continuous vars  <br/>
* MLE determines the regression coefficient that accurately predicts the probability of the binary dependent variable <br/>

#### Assumptions
* Observations are independent of each other
* No multicollinearity among independent vars
* Assumes a linear relationship between link function and independent variables 
* Dependent Variable is not normally distributed
* Errors need to be independent but not normally distributed
* Homoscedasticity is not required
* Dependent variable in logistic regression is not measured on an interval or ratio scale

#### Advantages
* It's fast, highly interpretable, doesn't require input features to be scaled, doesn't require any tuning, easy to regularize, and outputs are well-calibrated predicted probabilities
* Dependent Variable is not normally distributed




