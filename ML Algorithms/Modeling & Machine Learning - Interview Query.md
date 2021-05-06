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




















