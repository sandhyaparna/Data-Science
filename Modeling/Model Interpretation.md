### Links
https://towardsdatascience.com/human-interpretable-machine-learning-part-1-the-need-and-importance-of-model-interpretation-2ed758f5f476
https://www.kdnuggets.com/2018/12/explainable-ai-model-interpretation-strategies.html/2
https://towardsdatascience.com/explainable-artificial-intelligence-part-3-hands-on-machine-learning-model-interpretation-e8ebe5afc608



### Model Interpretation
When comparing models, besides model performance, a model is said to have a better interpretability than another model if its decisions are easier to understand by a human than the decisions from the other model.
* Model performance is not the run-time or execution performance, but how accurate the model can be in making decisions.
* What drives model predictions? Which features are imp in decision-making policies of the model - ensures FAIRNESS of the model.
* Why did the model take a certain decision? Validate and justify why certain key features were responsible in driving certain decisions taken by a model during predictions - ensures ACCOUNTABILITY and RELIABILITY of the model.
* How can we trust model predictions? We should be able to evaluate and validate any data point and how a model takes decisions on it - ensures TRANSPARENCY of the model.

* Global Interpretation - Being able to explain the conditional interaction between dependent andindependent variables based on the complete dataset
* Local Interpretation - Being able to explain the conditional interaction between dependent andindependent variables wrt to a single prediction

Traditional techniques for Model interpretation are 
* Exploratory analysis and visualization techniques like clustering and dimensionality reduction.
* Model performance evaluation metrics like precision, recall, accuracy, ROC curve and the AUC (for classification models) and the coefficient of determination (R-square), root mean-square error, mean absolute error (for regression models)


### Techniques
* Using Interpretable Models - DT, Linear Reg, Logistic reg, Naive Bayes, KNN
* Feature Importance using SHAP, Skater
* Partial Dependence Plots using SHAP, Skater
* Global Surrogate Models
* Skater, Python framework
* ELI5
* SHAP
* LIME - Local Interpretable Model-Agnostic Explanations

Concept behind global interpretations of model-agnostic feature importance
* We measure a feature’s importance by calculating the increase of the model’s prediction error after perturbing the feature.
* A feature is “important” if perturbing its values increases the model error, because the model relied on the feature for the prediction.
* A feature is “unimportant” if perturbing its values keeps the model error unchanged, because the model basically ignored the feature for the prediction.

### Skater
Frameworks like Skater compute Feature importance based on an information theoretic criteria, measuring the entropy in the change of predictions, given a perturbation of a given feature. The intuition is that the more a model’s decision criteria depend on a feature, the more we’ll see predictions change as a function of perturbing a feature. 

### SHAP
frameworks like SHAP, use a combination of feature contributions and game theory to come up with SHAP values. Then, it computes the global feature importance by taking the average of the SHAP value magnitudes across the dataset. 

### Global Surrogate Models






