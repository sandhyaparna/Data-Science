https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/
https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/ - Imablanced classification solutions for various bagging & boosing algos

### Multi-Classifiers
A set of weak learners (Train multiple models) are combined to create a strong learner that obtains better performance than a single one. <br/>
Helps in minimizing noise, bias and variance. <br/>
##### Ensemble methods - Same Learning algorithm - Bagging & Boosting
* N sets are created from Training data by random sampling with replacement - Each set is trained using same learning algorithm and thus N learners are generated
* Decreases the variance of your single estimate as they combine several estimates from different models. So the result may be a model with higher stability.
##### Hybrid methods - Different Learning algorithms - Stacking
Each learner uses a subset of dat
Output of primary classifiers, called level 0 models, will be used as attributes for another classifier(Combiner), called meta-model, to approximate the same classification problem. <br/>

### Advantages of Ensembles
* Ensembling is a proven method for improving the accuracy of the model and works in most of the cases.
* It is the key ingredient for winning almost all of the machine learning hackathons.
* Ensembling makes the model more robust and stable thus ensuring decent performance on the test cases in most scenarios.
* You can use ensembling to capture linear and simple as well non-linear complex relationships in the data. This can be done by using two different models and forming an ensemble of two.

### Disadvantages of Ensembles
* Ensembling reduces the model interpretability and makes it very difficult to draw any crucial business insights at the end.
* It is time-consuming and thus might not be the best idea for real-time applications.
* The selection of models for creating an ensemble is an art which is really hard to master.

### Bagging
* Bagging is best used when singe-model is over-fitting (High Variance). Boosting is not used when single-model is over-fitting as boosting itself suffers from over-fitting problem.
* Bagging reduced variance of the classifier, doesn't help much with bias.
* N sets are created from Training data by random sampling with replacement - Any element has the same probability to appear in a new data set.
* Training is parallel i.e. each model is built independently.
* Result is obtained by averaging the responses of the N learners (or majority vote).
* Works best when classifier is unstable (Decision Trees), where as bagging can hurt stable model by introducing artificial variability 
* In noisy data environments bagging outperforms boosting
##### Algorithms
* Random Forests - Parallel processing is not possible - https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
* Bagging meta-estimator - https://www.analyticsvidhya.com/blog/2018/02/introductory-guide-regularized-greedy-forests-rgf-python/
* Regularized Greedy Forests

### Boosting
* Boosting is best used when single model gets a very low performance (Model is too simple implies high bias i.e. each of the weak hypotheses has accuracy just a little bit better than random guessing) as it generates a combined model with lower errors as it optimises the advantages and reduces pitfalls of the single model (reduce Bias).
* N sets are created from Training data by random sampling with replacement - But observations are weighted and therefore some of them will take part in the new sets more often.
* Every time a new learner is built in a sequential way (outcome of one model becomes input to the next model), takes into account the previous classifiers’ success. After each training step, the weights are redistributed. Misclassified data increases its weights to emphasise the most difficult cases. In this way, subsequent learners will focus on them during their training.
* Boosting assigns a second set of weights, this time for the N classifiers, the algorithm allocates weights to each resulting model, a learner with good a classification result on the training data will be assigned a higher weight than a poor one, in order to take a weighted average of their estimates.
* In AdaBoost, an error less than 50% is required to maintain the model; otherwise, the iteration is repeated until achieving a learner better than a random guess.
##### Algorithms
* GBM
* Light GBM
* XGBoost - https://www.kdnuggets.com/2017/10/xgboost-top-machine-learning-method-kaggle-explained.html
https://www.kdnuggets.com/2017/10/understanding-machine-learning-algorithms.html
https://www.kdnuggets.com/2017/10/xgboost-concise-technical-overview.html
https://jessesw.com/XG-Boost/
* H2O
* LPBoost
* AdaBoost
* LogitBoost
* CatBoost - https://www.kdnuggets.com/2018/11/mastering-new-generation-gradient-boosting.html
https://www.kdnuggets.com/2018/03/catboost-vs-light-gbm-vs-xgboost.html



 <br/>
