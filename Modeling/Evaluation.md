### Links
Evaluation Metrics https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/ <br/>

### Healthcare
In healthcare imprecise predictions such as False-positives can overwhelm physicians, nurses, and other providers with false alarms. whereas False-negative predictions can miss significant numbers of clinically important events, leading to poor clinical outcomes.

• Micro-F1 score metric is used when a prediction has more than a single outcome

Underfitting <br/>
When Model is too simple, both training and test errors are large <br/>

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









