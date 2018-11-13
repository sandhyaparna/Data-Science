Naive Bayes is a classification technique based on Bayes' theorem with an assumption of independence among features.
Conditional Probability - 

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/09/Bayes_rule-300x172.png)

![](https://cdn-images-1.medium.com/max/800/1*1hE-O8DML8hmTf0DlD2AVw.gif)

### Assumptions
* Predictors are independent

### Pros
* It is easy and fast to predict class of test data set. It also perform well in multi class prediction.
* When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.
* It can be trained on small datset.
* It perform well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).
* It is not sensitive to irrelevant features.

### Cons
* If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as “Zero Frequency”. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.
* On the other side naive Bayes is also known as a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
* Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.

### Applications
* Real_time prediction - Since it is fast
* Text classification/Spam Filtering/Sentiment Analysis
* Recommendation System - Along with Collaborative filtering
