### Links
https://www.saedsayad.com/logistic_regression.htm <br/>
https://www.kdnuggets.com/2019/01/logistic-regression-concise-technical-overview.html <br/>

https://towardsdatascience.com/logistic-regression-explained-9ee73cede081 </br>
In logistic regression, linear combination of inputs are mapped to log odds. It predicts the probability of occurrence of an event by fitting data to a logic function.  <br/>
logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk <br/>
* probability is expressed as sigmoid function i.e 1/(1+e^-x)
* So, log( p/(1-p) ) can be expressed as linear combination of inputs

### Interpretation of Categorical and Continuous vars of a Logistic Reg
Odds Ratio = p/(1-p)
* For each predictor u get p-value and OddsRatio=exp(Xi). Xi is coeff. Odds ratio gre than 1 implies positive relation
* https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
  * Female variable is created from gender var. In female var 1=female, 0=male; if coeff of this var is 0.593 i.e log(p/(1-p)) or log of odds ratio is 0.593; odd ratio = e^0.593 i.e 1.809 implies odds for female are 80.9% higher than the odds for males
  * For a continuous var, if coeff of the var i.e log of odds ratio is 0.1563, implies odds ratio is e^0.1563 = 1.169 implies if there is 1 unit inc in the continuous var, we see 16.9% inc in the odds of being Target=1
  * Logistic regression with multiple predictor variables: This fitted model says that, holding math and reading at a fixed value, the odds of getting into an honors class for females (female = 1)over the odds of getting into an honors class for males (female = 0) is exp(.979948) = 2.66.  In terms of percent change, we can say that the odds for females are 166% higher than the odds for males.  The coefficient for math says that, holding female and reading at a fixed value, we will see 13% increase in the odds of getting into an honors class for a one-unit increase in math score since exp(.1229589) = 1.13. </br>
         hon |      Coef </br> 
        math |   .1229589   </br>
      female |    .979948   </br>
        read |   .0590632   </br>
   intercept |  -11.77025 </br>
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
* If Loss function derivative is +ve then move towards left  implies decrease weights and if the slope value is big then take a big stepsize as slope is big (more change in y for change in x) when it is farther from destination(global min)
* If Loss function derivative is -ve then move towards right implies increase weights and if the value is small then take a small stepsize because slope is small nearer to the global min
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
https://towardsdatascience.com/probability-learning-iii-maximum-likelihood-e78d5ebea80c </br>
https://www.analyticsvidhya.com/blog/2018/07/introductory-guide-maximum-likelihood-estimation-case-study-r/ </br>
* MLE is used to find the coefficients (estimators) that accurately predicts the probability of the binary dependent variable, that minimizes the likelihood function
* MLE can be defined as a method for estimating population parameters (such as the mean and variance for Normal, number of trails & probability of success in an individual trail for binomial distribution, rate (lambda) i.e which is the expected number of events in the interval (events/interval * interval length) and the highest probability number of events) from sample data such that the probability (likelihood) of obtaining the observed data is maximized.
* The goal of maximum likelihood is to fit an optimal statistical (normal, exponential, poisson) distribution to some data
  * Normal / Gaussian distribution
  * Binomial distribution:
When a trail of two outcomes (as success and fail) is repeated for n times and when the probabilities of “number of success event” is logged, the resultant distribution is called a binomial distribution. For an example lets toss a coin for 10 times (n = 10) and the success is getting head. So if we log the probabilities of getting head only one times, two times, three times, … then that distribution of probabilities is in a binomial distribution.
  * Poisson distribution: https://towardsdatascience.com/the-poisson-distribution-and-poisson-process-explained-4e2cb17d459#:~:text=The%20Poisson%20distribution%20is%20defined,highest%20probability%20number%20of%20events.&text=Even%20if%20we%20arrive%20at,the%20average%20time%20between%20events. lambda is avg no of events happening in a given tim e period. graph shows the prob of given number of events happening. foe eg when avg no of events happening is 5, p(4) or P(6) happening is a little less, P(3) or P(7) is happening is more less
  * Eg: if we need classify height as either male or female. we fit a statibinomial distributionstical distribution to Female data and another statistical distribution to Male data. when we a get a new value of height, we find the probability of new point belonging to either of the distributions and assign to the gender than has higher probability. 
  * Parameters are mean and variance of the statistical distributions that we are fitting. we can say we are looking for a curve that maximizes the probability of our data given a set of curve parameters.
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

