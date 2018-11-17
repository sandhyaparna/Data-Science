### Multi-Classifiers
A set of weak learners (Train multiple models) are combined to create a strong learner that obtains better performance than a single one. <br/>
Helps in minimizing noise, bias and variance. <br/>
##### Ensemble methods - Same Learning algorithm - Bagging & Boosting
* N sets are created from Training data by random sampling with replacement - Thus N learners are generated
##### Hybrid methods - Different Learning algorithms - Stacking
Output of primary classifiers, called level 0 models, will be used as attributes for another classifier, called meta-model, to approximate the same classification problem. <br/>

### Bagging
* N sets are created from Training data by random sampling with replacement - Any element has the same probability to appear in a new data set.

### Boosting
* N sets are created from Training data by random sampling with replacement - But observations are weighted and therefore some of them will take part in the new sets more often.



 <br/>
