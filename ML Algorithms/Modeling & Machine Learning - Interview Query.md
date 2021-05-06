## Intro
Types of Machine Learning Interview Questions
* Modeling Case Study - Experience related project, etc
  * Describe how you would build a model to predict Uber ETAs after a rider requests a ride
  * How would you evaluate the predictions of an Uber ETA model?
  * What features would you use to predict the Uber ETA for ride requests?
* Recommendation and Search Engines
  * How would you build a recommendation engine to recommend news to users on Google?
  * How would you evaluate a new search engine that your co-worker built?
* ML Concepts
* Applied Modeling -  require more contextual knowledge and information gathering from the interviewer
  * You’re given a model with 90% accuracy, should you deploy it?
  
  
## Modeling Case Study
* Data Exploration & Pre-Processing
* Feature Selection & Engineering
* Model Selection
* Cross Validation
* Evaluation Metrics
* Testing and Roll Out

1. Clarification - cohort of interest, Data, Imbalance or not, chosse the evaluation metric based on the class, imbalance, regression
The most common mistake that most people make when facing a modeling case study question is to underestimate the scope of the problem. For example, many inexperienced candidates will dive right into selecting a model.
Let’s take an example question asked by Uber: How would you build a model to give an estimated time of arrival for selecting a ride?
* Bad Response: 
  * I would build a regression model and use features like time of day, passenger location, and driver distance. I would select a neural network architecture that would work well….
* Good Response
  * What’s the scope of this model? Is it for a particular city or location, across the entire US, or for the entire world?
  * Additionally, how many ride ETAs would we have to serve per minute?
  * Lastly, how much does accuracy matter? Do riders have a propensity to cancel their rides if the estimated time is much longer than the actual estimated time after matching?
* We have to understand a few things right off the bat when presented with a case study.
  * Why would we want to build this model in the first place?
  * What is the model is being used for?
  * How would the model be integrated into the product?
  * How does the model affect the business?
  * How much do we care about model accuracy?
2. Make assumptions
The model should be used across the entire United States and should be able to serve any user that wants a ride at any given time. I am assuming that the model would serve a prediction when a rider opens the Uber app and then immediately sees a ride estimated ETA if the rider were to request a ride.
3. Define Priorities
There is the consideration that the lower the ETA our model provides, the higher the propensity of a user to request a ride at that very moment. However, I’ll assume that our goal is to address complete accuracy of the ETA and let another ensemble of models address what should be done to increase ride request rates. </br>
Lastly, I want to dive into the specifics of how this model could be built. I am going to assume that there is already an existing ETA model that serves predictions currently in production. I’m also assuming we have location data of both the user and drivers at any point in time while they are active on the app, and we are saving the actual time of arrival data, which is computed by end time minus start time. End time is defined by the time that the driver arrives to pick up the user and start time is defined by when the user presses the request button.
Our only added points are that we want to make sure:
* Whether we are building a new model or improving upon an existing one.
* Whether the user behavior based on the effect of ETAs is out of scope or not.
* What exists in our dataset to build this model.


## Data Exploration & Pre-Processing
* Dealing with missing data
  * How would we deal with missing data if 1% of the features were missing?
  * What about if 50% of the features are missing?
* skewed variables
* outliers: Outliers consist of two different types: univariate and multivariate 
* class imbalance 
  * Why is the target variable imbalanced?
  * What is the degree of the imbalance? Is it 80⁄20 or 99⁄1? - 
  * What are the costs of misclassification?
* Skewed data of Continuous target: Skewed data can apply to the target variable or feature variables
  * What do we do if the target distribution is left-skewed? What about right-skewed?
  * How do we measure the skewness of our distribution?

#### Bank Fraud Model
* Let's say that you work at a bank that wants to build a model to detect fraud on the platform. The bank wants to implement a text messaging service in addition that will text customers when the model detects a fraudulent transaction in order for the customer to approve or deny the transaction with a text response. How would we build this model?
* My ideas: Binary class model; Is the cohort of interest: credit cards, debit cards, cash transactions or withdrawls; Imbalance nature; AUC, precision-recall curves
* Link: https://www.interviewquery.com/questions/bank-fraud-model

#### Missing housing data
* We want to build a model to predict housing prices in the city of Seattle. We've scraped 100K sold listings over the past three years but found that around 20% of the listings are missing square footage data.How do we deal with the missing data to construct our model?
* My ideas: Does only this variable have missing values? We can use KNN imputation based on number of beds, bath, garage area, etc. Or if there is any description variable, we can extract from the text or If we have photos, may be can just impute based on the number of photos assuming no of photos is proportional to sq foot size
* Link: https://www.interviewquery.com/questions/missing-housing-data
  * Ignore observations with missing values, Median imputation, Median imputation based on categorical groups of number of beds/bathrooms etc
  
#### Variable Error
* Assume you have a logistic model that is heavily weighted on one variable and that one variable has sample data like 50.00, 100.00, 40.00, etc.... Next let's assume that there was a data quality issue with that variable and an unknown number of values removed the decimal point. For example 100.00 turned into 10000. Would the model still be valid? Why or why not? How would you fix the model?
* My ideas: Model will not be valid. If all the values in the variable have same number of decimal values, may be scaling or normalization works
* Link: https://www.interviewquery.com/questions/variable-error


## Feature Selection
* Rider Demographics, Rider Activity, Driver Activity, Location Characteristics, Ride Characteristics

#### Keyword Bidding
* Let's say you're working on keyword bidding optimization. You're given a dataset with two columns. One column contains the keywords that are being bid against, and the other column contains the price that's being paid for those keywords.Given this dataset, how would you build a model to bid on a new unseen keyword?
* Is it just 1 word or a group of words in a observation
* Link: https://www.interviewquery.com/questions/keyword-bidding

#### Multicollinearity in Regression
* Link: https://www.interviewquery.com/questions/multicollinearity-in-regression

## Model Selection - Explainability, features count, observations count, categorical vs Numerical, Linear model vs Non-linear, Training speed, Prediction speed & performance
* Trade-off between explainability and performance / accuracy
* Link: https://www.interviewquery.com/course/data-science-course/lessons/modeling

#### Booking Regression
* Let's say we want to build a model to predict booking prices on Airbnb. Between linear regression and random forest regression, which model would perform better and why?
* My ideas: If we are trying to compare between linear and bagged non-linear model. We will be interested in understanding if the assumptions of linear regression satisfy or not, i.e Target is having linear relationship with Independent vars; no multicollinearity, which we can work during feature selection; no outliers which depends on the data that we have. We can try Decision tree model, as it captures non-linearity and if the model is overfitting, we can try random forest as it is doing bagging 
* Link: https://www.interviewquery.com/questions/booking-regression

#### Evaluate News
* Let’s say you are given a model that predicts whether a piece of news is relevant or not when shared on Twitter. How would you evaluate the model?
* Link: https://www.interviewquery.com/questions/evaluate-news


## Machine Learning Algorithms Framework
* particular machine learning algorithm: 
  * What is the big idea of [ML TECHNIQUE]?
  * Explain step by step how [ML TECHNIQUE] works.
* Intutive Explanation
  * What is the intuition behind [ML TECHNIQUE]?
  * How would you explain this technique to a non-technical person?
* Difference Between Algorithms
  * How would you compare [INSERT TECHNIQUE] with a [INSERT SIMILAR TECHNIQUE]?
  * When would you use [ML TECHNIQUE 1] versus [ML TECHNIQUE 2]?
* Assumptions
  * What assumptions does [INSERT TECHNIQUE] make?
* Tuning and Parameters
  * What are [INSERT TECHNIQUE]’s parameters and how would you tune them?
* Pros and Cons
  * What are [INSERT TECHNIQUE]’s pros and cons?
  * For example, let’s take a look at Naive Bayes. What are the pros/cons of using Naive Bayes?
    * Pros:
      * Fast predictions on test sets.
      * Works well for multiple classes.
      * Requires less training data.
      * Good interpretability.
    * Cons:
      * If a class was not observed in training data, the class will have a zero value.
      * Predicted probabilities are not reliable estimators.
      * Strong assumptions around normality.
* Performance
  * When would [INSERT TECHNIQUE] fail to perform well?
  * When would [INSERT TECHNIQUE] perform well?
  * Similar to evaluating pros and cons, we can try to imagine scenarios where a specific algorithm would perform well and scenarios where they would not. For example, if we take linear regression:
    * Linear regression would perform well if we were in a scenario where we’d want to understand the coefficients of different features and how they relate to the predictor class. Let’s say we wanted to know how neighborhoods affect housing prices in a new city. We can use regression to get a numerical coefficient representing the relative affordability of each neighborhood.
    * Linear regression would not perform well if any of its assumptions are violated, or if we’re given many more categorical features versus continuous ones.
* Optimization Speed
  * What is [INSERT TECHNIQUE]’s optimization speed?
* Special Features
  * What are the special cases/features of [INSERT TECHNIQUE]?


## Regression, Regularization, and Gradient Descent
#### Regression
* Interpret Linear regression coefficients: The regression coefficient signifies how much the mean of the dependent variable changes, given a one-unit shift in that variable, holding all variables constant.
* Interpret logistic regression coefficients: The assumption is that the log transform of our target has a linear relationship with our predictor variables. To convert log odds to odds, we must apply an exponential transformation and then convert the odds ratio into a probability. Once we do this, the coefficients become much easier to understand. 
* Maximum Likelihood Estimation: Maximum Likelihood Estimation is where we find the distribution that is most likely to have generated the data. To do this, we have to estimate the parameter theta, that maximizes the the likelihood function evaluated at x. P(data | X).  MLE means we are aiming to find the best parameter that maximize the likelihood of the distribution. It happens, that the MLE for linear regression is the same as OLS
  * The likelihood function measures how likely you would be to observe the data you have for various values of the parameters of your model.

#### Gradient Descent
* Gradient Descent: Gradient descent is a method of minimizing the cost function. The form of the cost function will depend on the type of supervised model. When optimizing our cost function, we compute the gradient to find the direction of steepest ascent. To find the minimum, we need to continuously update our Beta, proportional to the steps of the steepest gradient.
  * What is the learning rate in gradient descent? The learning rate controls the size of the step our algorithm takes on each iteration.
  * What happens if our learning rate is too small? Our algorithm will take too long to converge (though in principle, it should eventually converge.)
  * What happens if our learning rate is too large? The algorithm will diverge.
  * How do you know when to stop iterating within gradient descent? You hit the threshold parameter that checks the size of each step that’s taken. If the step size is smaller, then we stop iterating. You hit the maximum number of iterations. This is set manually.
  * If gradient descent fails to converge, what could you do? Reduce the learning rate and change the schedule, which means, tune the way the learning rate gets updated. Typically, this is from a larger learning rate to a smaller learning rate.
* Stochastic Gradient Descent: hastic Gradient Descent is a faster form of Gradient Descent. Rather than use all the data points to minimize the cost function, we select a single sample and use it to update our regression coefficients.
  * What is the benefit of using stochastic gradient descent over standard gradient descent? If our dataset consists of millions or more samples, we may be unable to load the entire dataset into memory at once. This may necessitate us to multiple large matrices each time or accumulate the effects of every sample on the cost function before moving onto the next iteration.
  * How would you pick the stopping criterion for Stochastic Gradient Descent? There’s no perfect solution here. One was is to keep a holdout set and stop when the cost function on that set begins to increase. Repeat this for multiple holdout sets to have stronger confidence.
* Batch Gradient Descent: Batch gradient descent is the happy medium between gradient descent and stochastic gradient descent. Rather than use all the data points or a single sample, we use a subset of data to update our coefficients.
  * What is the benefit of using batch gradient descent? Batch gradient descent can overcome the issue of statistical fluctuations encountered in stochastic gradient descent. On the other hand, we’re not using all the data points, so we avoid the slowness of regular gradient descent.
  * How do you determine the batch size in batch gradient descent? Generally, we want to pick the sample sizes where we can benefit from the effects of CLT. This is usually n > 30.

#### Regularization
* Regularization: Regularization is the act of modifying our objective function by adding a penalty term, to reduce overfitting
* L1 and L2 Regularization: L1 and L2 regularization are methods used to reduce the overfitting of training data. Least Squares minimizes the sum of the squared residuals, which can result in low bias, but high variance. L2 is less robust, but has a stable solution and always one solution. L1 is more robust but has an unstable solution, and can possibly have multiple solutions.
* Lasso: If you take the ridge regression penalty and replace it with the absolute value of the slope, then you get Lasso regression or L1 regularization.
  * Pros:
    * Performs feature selection automatically.
    * Ability to remove correlated variables.
  * Cons:
    * Instability. If you change the regularization parameter, the non-zero coefficients can change substantially in any direction. Coefficients can change drastically.
    * Performs erratically with high-dimensional data or when several features are strongly correlated.
* Ridge Regression: L2 Regularization, also called ridge regression, minimizes the sum of the squared residuals plus lambda times the slope squared. This additional term is called the Ridge Regression Penalty. This increases the bias of the model, making the fit worse on the training data, but also decreases the variance.
  * Pros:
    * Likely computationally faster than LASSO.
    * Can reduce variance, but increase bias.
    * Works better when OLS estimates have high variance.
  * Cons:
    * Trades variance for bias, this can become a negative.
    * Doesn’t necessarily remove collinearity.
    * Interpretability is difficult.
* In what scenarios would you expect lasso to perform better than ridge? In general, one might expect the lasso to perform better in a setting where a relatively small number of predictors have substantial coefficients, and the remaining predictors have coefficients that are very small or that equal zero. Ridge regression will perform better when the response is a function of many predictors, all with coefficients of roughly equal size.


## Tree Algorithms 
#### Decision Tree
A decision tree uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.
* How does a decision tree split on continuous variables? A decision tree will look at all the data [1,2,3,4] and then split on the midpoints between each number: [1.5,2.5,3.5]. Then, it’ll determine the information gain or reduction in RSS for that particular split to decide on the best split.
* How does a decision tree split on nodes? By using impurity. Impurity is a measure of the homogeneity of the labels on a node. Information gain uses the entropy measure as the impurity measure and splits a node such that it gives the most amount of information gain.
* Link: http://www.acheronanalytics.com/acheron-blog/brilliant-explanation-of-a-decision-tree-algorithms

#### Random Forest
Random forest is a supervised learning algorithm. The forest is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.
* How does a Random Forest deal with correlated features? Set the feature fraction to be smaller, in order to reduce the likelihood of correlated features having an affect on the model. But let’s say, that we set the max features to 100%. What would happen? Correlation should have as large of an effect, compared to linear models, since the decision trees will only pick one feature to split on. However, if there are collinear features, it means the tree is much more likely to split on the correlated features. If by chance, it splits on the correlated features more often, it could affect results, but the likelihood is low.
* What happens if we have correlated trees in a Random Forest? We basically lose the benefit of a Random Forest. This essentially becomes a model with many decision trees that share the same features. This is likely to increase the bias of our model since the decision trees are likely to bias towards splitting on similar values.
* Explain how feature importances are calculated. Feature importance is calculated by looking at the total amount of information gain for each feature. Then we average all the information gain across all the trees.
* What is out of bag error? OOB is a method of validating a random forest model. When we bootstrap, we’re only selecting a sub-sample of the dataset. In this case, the data we do not select would be our out of bag data. The out of bag error is then calculated using a subset of DT that do not contain the OOB sample. This is very similar to LOOCV.
* What types of models does bagging typically work well for? Models with high-variance and low-bias typically work well with bagging. Bagging will reduce the variance of our predictions. We don’t worry too much about pruning, because the averaging will reduce the variance.
* When can OOB score by useful? In scenarios where we do not have too much data and want to use the entire dataset for training.

#### Gradient Boosted Trees
* How do boosted trees handle multicollinearity?
* How would you regularize a gradient boosted tree?
* How would you compare XGBoost vs. LightGBM?
* What are the pros/cons of using boosted trees?
* Link: https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/


## Model Evaluation
#### Cross Validation
* K-Fold Cross Validation: split data
* We can also use cross-validation to select the best parameters for our model. We would then train the model on different parameters and use cross validation to select the highest score.
* When would we use K-Fold cross-validation?Depending on the size of the dataset, we’ll need a large number of data points to perform KFold cross validation.
Computationally it will take longer to train and evaluate models using K-Fold cross validation. If computational resources are not available, then it makes sense to do a regular train-test-split.
* Deciding on the number of folds of K-Fold cross validation? This part is tricky but setting a standard number that optimizes computational time and performance will be key. A larger K means less bias towards overestimating the true expected error but leads to higher variance and higher running time. Notably, as K gets larger, the method starts turning into Leave-one-out cross validation.
* How would we validate that the distributions of our training set and testing set are roughly the same?
  * Check whether the mean, standard deviation of the target values, and each feature value in train/test are roughly the same.
  * Build a model that predicts whether a datapoint is in train or test. Measure the ROC-AUC to see if it’s above 0.5.
* Let’s say that we find that our train/test set distributions differ. What would we do to rebalance our distributions?
  * We could implement Stratified K-Fold cross validation. This will reduce the likelihood we chose a poor split.
  * Dropping high propensity score feature importances. Remove the highest-scoring features within the propensity model as it signals the largest covariate shifts.
  * Re-weight your dataset through over-sampling, weighting so the train/test set distributions match.

#### Evaluation Matrics
* Evaluation Metrics - Questions to Understand
* Decide where you want to set your focus: False Alarms (Bombs) vs Missing out (Loan)
* Calculate the metrics; Do not forget to take a look at the other scores as well, in order to prevent the model to fool you
* Plot a Confusion matrix; Does the diagonal contain high values?
* If needed go for more Advanced Metrics like ROC, Precision Recall Curve for imbalanced Data, etc..
* How would you compare F1 score with PR-AUC?
* How would you compare F1 score with PR-AUC for evaluating imbalanced datasets?
* How would you compare using MAE vs. RMSE?
* How would you compare using R2 vs. MAE/RMSE?
* How would you compare using PR-AUC vs. ROC-AUC?
* How would you compare ROC-AUC vs. Accuracy?
* How would you compare Precision vs. Recall?
* How would you deal with a scenario where different classes have different importance?
* What evaluation metric would you use if you cared about under-predictions?
* What evaluation metric would you use if you cared about over-predictions?
* Link:https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc#4

#### Testing and Roll Out
* Setting a baseline
* Backtesting and Validation
* AB Testing Roll Out


## Recommendation and Search Engines
#### Case Studies
* How would you build Facebook’s newsfeed recommendation engine?
* How would you compare two different search engines?

#### System Design* 
* How do you scale a recommendation engine to millions of users on Youtube?
* How do you build a search engine that can return results within a few milliseconds?
* How would you take a collaborative filtering algorithm and implement it in production?

#### Algorithms
* Unsupervised:KNN or Collaborative filtering
* 

#### Model Evaluation
* Three things to consider: Execution of the experiment, then metrics to track in product, and lastly the manual rated relevancy.
* Run an A/B test to judge the preliminary results
* Metrics for reco engine:
  * click through rate and clicks per search:  would give us values that directly impact revenue that cannot go down by a certain X percentage compared to the control
  * Session abandonment rate: The calculation of the number of search sessions that do not end in any search result clicked.
  * Session success rate: The calculation of the number of search sessions that lead to success divided by the total search sessions. Success can be defined in a variety of ways but we can assume it is when the user has received the answer to their initial search query. This can be calculated by proxy metrics such as dwell time on a search result url or copy pasting urls that they have found useful.
  * Zero result rate: The calculation of the number of results returned with zero search results.
* Offline metrics
  * Precision and recall are some of the most standardized ways to evaluate search. We can define precision as the fraction of the documents retrieved that are relevant to the user's information need and recall as the fraction of the documents that are relevant to the query that are successfully retrieved. Within precision there is also:
    * Precision at N: This is taking the precision value at a cutoff of N documents retrieved. For example if we look at the number of relevant documents within the first 10, first 20, and first 30 documents, we can compare the ranking of our search engine in terms of relevancy within the first 10, 20, and 30 documents returned.
    * Average Precision: If we can compute an average precision at every position in the ranked sequence of documents returned, we can plot a precision recall curve. Given our search engine will return a ranked sequence of documents, this will help is further compare our precision recall metrics.
* Manual metrics
  * We can test the algorithm offline, benchmarking how the results rank with the new algorithms and if the URLs are higher quality than the previous algorithms in place. The quality is based on how the search quality raters rate the URLs in previous cases. If the URLs were unrated, we can request these raters to rate the new URLs or compare the old search results to this new test set.
  * Live tests where we sample a subset of real live searchers and give them the new results with the new set of test algorithms. If we see a higher click through rate on the new search results, it may imply that the new results are better than the older ones. 

#### Job Recommendation
* Link: https://www.interviewquery.com/questions/job-recommendation







