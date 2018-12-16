### Links
https://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html
https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html


### Techniques
* Do Nothing
* Sampling (CROSS-VALIDATION should always be applied before over sampling to avoid over-fitting)
  * Oversample the minority class - SMOTE
  * Undersample the majority class.
  * Synthesize new minority classes.
  * Build n models that use all the samples of the rare class and n-differing samples of the abundant class. Eg: Given that you want to ensemble 10 models, you would keep e.g. the 1.000 cases of the rare class and randomly sample 10.000 cases of the abundant class. Then you just split the 10.000 cases in 10 chunks and train 10 different models.
* Anomaly Detection
* Adjust the decision threshold.
* Adjust the class weight (misclassification costs).
* Modify an existing algorithm to be more sensitive to rare classes.

### Evaluation Metrics
* Always test(Test set) on original distribution 
* ROC-AUC, Precision-Recall, Lift or Gain curves, F1 score, MCC(Corr coeff between observed & predicted binary classifications.
* Decision threshold on predict_proba
* 
* 
* 
* 
* 




