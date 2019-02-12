### Links
https://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html <br/>
https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html <br/>
https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/ <br/>
https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/ <br/>

Test set should never be undersampled or over-sampled <br/>

### Disadvantages of Imbalanced data set:
A few classifier algorithms have bias towards classes that have more number of instances.They tend to only predict the majority class data.
The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.

### Techniques
* Do Nothing and use models that can handle imbalance like XGBoost
* Sampling (CROSS-VALIDATION should always be applied before over sampling to avoid over-fitting)
  * Oversample the minority class - SMOTE
  * Undersample the majority class.
  * Synthesize new minority classes - ROSE & DMwR packages in R.
  * Build n models that use all the samples of the rare class and n-differing samples of the abundant class. Eg: Given that you want to ensemble 10 models, you would keep e.g. the 1.000 cases of the rare class and randomly sample 10.000 cases of the abundant class. Then you just split the 10.000 cases in 10 chunks and train 10 different models.
  * Above appraoch of ensembling can be fine-tuned by playing with the ratio between the rare and the abundant class. The best ratio  heavily depends on the data and the models that are used. But instead of training all models with the same ratio in the ensemble, it is worth trying to ensemble different ratios.  So if 10 models are trained, it might make sense to have a model that has a ratio of 1:1 (rare:abundant) and another one with 1:3, or even 2:1.
* Anomaly Detection
* Adjust the decision threshold.
* Adjust the class weight (misclassification costs).
* Modify an existing algorithm to be more sensitive to rare classes.
* Cluster the abundant class - Instead of relying on random samples to cover the variety of the training samples, he suggests clustering the abundant class in r groups, with r being the number of cases in r. For each group, only the medoid (centre of cluster) is kept. The model is then trained with the rare class and the medoids only.

### Evaluation Metrics
* Always test(Test set) on original distribution 
* ROC-AUC, Precision-Recall, Lift or Gain curves, F1 score, MCC(Corr coeff between observed & predicted binary classifications.
* Decision threshold on predict_proba
* 
* 
* 
* 
* 




