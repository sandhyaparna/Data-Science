### Multi-Classifiers
A set of weak learners (Train multiple models) are combined to create a strong learner that obtains better performance than a single one. <br/>
Helps in minimizing noise, bias and variance. <br/>
##### Ensemble methods - Same Learning algorithm - Bagging & Boosting
* N sets are created from Training data by random sampling with replacement - Each set is trained using same learning algorithm and thus N learners are generated
* Decreases the variance of your single estimate as they combine several estimates from different models. So the result may be a model with higher stability.
##### Hybrid methods - Different Learning algorithms - Stacking
Output of primary classifiers, called level 0 models, will be used as attributes for another classifier, called meta-model, to approximate the same classification problem. <br/>

### Bagging
* Bagging is best used when singe-model is over-fitting. Boosting is not used when single-model is over-fitting as boosting itself suffers from over-fitting problem.
* N sets are created from Training data by random sampling with replacement - Any element has the same probability to appear in a new data set.
* Training is parallel i.e. each model is built independently.
* Result is obtained by averaging the responses of the N learners (or majority vote).

### Boosting
* Boosting is best used when single model gets a very low performance (Model is too simple implies high bias) as it generates a combined model with lower errors as it optimises the advantages and reduces pitfalls of the single model (reduce Bias).
* N sets are created from Training data by random sampling with replacement - But observations are weighted and therefore some of them will take part in the new sets more often.
* Every time a new learner is built in a sequential way, takes into account the previous classifiers’ success. After each training step, the weights are redistributed. Misclassified data increases its weights to emphasise the most difficult cases. In this way, subsequent learners will focus on them during their training.
* Boosting assigns a second set of weights, this time for the N classifiers, the algorithm allocates weights to each resulting model, a learner with good a classification result on the training data will be assigned a higher weight than a poor one, in order to take a weighted average of their estimates.
* In AdaBoost, an error less than 50% is required to maintain the model; otherwise, the iteration is repeated until achieving a learner better than a random guess.
##### Algorithms
* GBM
* Light GBM
* XGBoost
* H2O
* LPBoost
* 
* CatBoost - https://www.kdnuggets.com/2018/11/mastering-new-generation-gradient-boosting.html




 <br/>
