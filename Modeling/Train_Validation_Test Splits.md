### Links
Train a new model on all available data https://machinelearningmastery.com/train-final-machine-learning-model/ <br/>
Train-Validation-Test https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7 <br/>
DONT split Train & Test based on Time https://discuss.analyticsvidhya.com/t/is-it-wise-to-split-training-and-test-dataset-based-on-time-year/2967/4 <br/>
Split data coming from DIFFERENT distributions https://www.kdnuggets.com/2019/01/when-your-training-testing-data-different-distributions.html <br/>
TEST nor VALIDATION set should never be undersampled or over-sampled. Only Training data should be over-sampled/under-sampled <br/>


### Avoid Common Errors 
* Lack of generalization - Create 3 identically distributed Machine Learning data sets such that it is repeatable
* Repeatable Datasets - use farm_fingerprint function on dates etc
https://cloud.google.com/bigquery/docs/reference/standard-sql/hash_functions <br/>
https://github.com/sandhyaparna/training-data-analyst/blob/master/courses/machine_learning/deepdive/02_generalization/repeatable_splitting.ipynb <br/>
https://github.com/sandhyaparna/training-data-analyst/blob/master/courses/machine_learning/deepdive/02_generalization/create_datasets.ipynb  <br/>
https://googlecoursera.qwiklabs.com/focuses/25429?locale=en <br/>
* Clean & Pre-proces data before splitting

### Splitting
* Split data into Train & Test - Stratified splits if the data is imbalanced
* Use Train data to perform cross-validation. A seperate Test set makes more sense when you use cross-validation or manully change the model hyperparameters based on the results of your model on the validation set. 
* Size of validation set within cross-validation is dependent on the overall data and model that you are training. Some models need substantial data to train upon, so in this case you would optimize for the larger training sets. Models with very few hyperparameters will be easy to validate and tune, so you can probably reduce the size of your validation set, but if your model has many hyperparameters, you would want to have a large validation set as well(although you should also consider cross validation). Also, if you happen to have a model with no hyperparameters or ones that cannot be easily tuned, you probably don’t need a validation set too!
* A final deployement model is built based on both Train & Test but we report only Test set performance to finally show how a model is performing




