### Links
Evaluation Metrics https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/ <br/>
https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc  <br/>
Cost Function https://towardsdatascience.com/model-performance-cost-functions-for-classification-models-a7b1b00ba60 <br/>
ROC-AUC https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5 <br/>
Loss functions 

### Confidence(1) for model output
* Decision tree: (probability/Confidence is number of 1s in that path) / (Total population in that path)
* Ensemble Models: Probabilities can be calculated by the proportion of decision trees which vote for each class

### Healthcare
* In healthcare imprecise predictions such as False-positives can overwhelm physicians, nurses, and other providers with false alarms. whereas False-negative predictions can miss significant numbers of clinically important events, leading to poor clinical outcomes.
* Micro-F1 score metric is used when a prediction has more than a single outcome
* In highly imbalanced data - confidence scores are less than 0.5 but they should be normalized to 0.5 or any other higher value
* Underfitting - When Model is too simple, both training and test errors are large <br/>

### Error Analysis on Validation set 
* A few examples of correct labels at random
* A few examples of incorrect labels at random
* Examples of the most accurate labels of each class (i.e., those with the highest probability that are correct)
* Examples of the most inaccurate labels in each class (i.e., those with the highest probability that are incorrect)
* Examples of the most uncertain labels (i.e., those with probability closest to 0.5)


### Model Validation from real-time predictions
* Prediction from Model - Cohort(Total) of patients that are scored as either 0(Predicted False) or 1(Predicted One) should be considered
* Confusion matrices can be created at different model score cutoffs
* True outcome from client should be considered as Actual True outcome
* If we get only Predicted by Model and Actual outcome, then we can calculate TP, FP, FN but not True Negatives
* For Validation analysis based on TOP - Min time of first time a threshold value is reached is considered for analysis

### Cost Function
* Cost is incurred for each predicted by model as cost is incurred for intervention
* Cost is the fine incurred for False Negatives

### Probability re-caliberation when doing under sampling
* https://towardsdatascience.com/probability-calibration-for-imbalanced-dataset-64af3730eaab
* Let ps be the probability of the prediction being a positive class after random undersampling (thershold should be varied and 0.5 is not ideal while calculated ps)
* p be the probability of the prediction given features 
* prob/(prob+(1-prob)/beta)
![](https://miro.medium.com/max/248/1*w-VK4WWmFxE5Gb25BhEY3g.png)
* where beta is observations where Target=1 / observations where Target=0

### Multi-label classification
* https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
* https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
* In multi-label classification, a misclassification is no longer a hard wrong or right. A prediction containing a subset of the actual classes should be considered better than a prediction that contains none of them, i.e., predicting two of the three labels correctly this is better than predicting no labels at all.
* Micro-averaging & Macro-averaging (Label based measures):
  * In micro-averaging all TPs, TNs, FPs and FNs for each class are summed up and then the average is taken.
  ![](https://miro.medium.com/max/617/1*nWbsBPAFl3WmU_bgtahqKQ.png)
  * In micro-averaging method, you sum up the individual true positives, false positives, and false negatives of the system for different sets and the apply them. And the micro-average F1-Score will be simply the harmonic mean of above two equations.
  * Macro-averaging is straight forward. We just take the average of the precision and recall of the system on different se
  ![](https://miro.medium.com/max/537/1*AwYON8c48oMm5AcqVxLiWQ.png)
  * Macro-averaging method can be used when you want to know how the system performs overall across the sets of data. You should not come up with any specific decision with this average. On the other hand, micro-averaging can be a useful measure when your dataset varies in size.
* Hamming-Loss (Example based measure) Ranges from 0 to 1. Less value implies better model
  * In simplest of terms, Hamming-Loss is the fraction of labels that are incorrectly predicted, i.e., the fraction of the wrong labels to the total number of labels.
* Exact Match Ratio (Subset accuracy):
  * It is the most strict metric, indicating the percentage of samples that have all their labels classified correctly.
  * There is a function in scikit-learn which implements subset accuracy, called as accuracy_score.




