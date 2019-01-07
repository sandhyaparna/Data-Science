### Links
https://towardsdatascience.com/human-interpretable-machine-learning-part-1-the-need-and-importance-of-model-interpretation-2ed758f5f476
https://www.kdnuggets.com/2018/12/explainable-ai-model-interpretation-strategies.html/2
https://towardsdatascience.com/explainable-artificial-intelligence-part-3-hands-on-machine-learning-model-interpretation-e8ebe5afc608



### Model Interpretation
When comparing models, besides model performance, a model is said to have a better interpretability than another model if its decisions are easier to understand by a human than the decisions from the other model.  <br/>
* Model performance is not the run-time or execution performance, but how accurate the model can be in making decisions.
* What drives model predictions? Which features are imp in decision-making policies of the model - ensures FAIRNESS of the model.
* Why did the model take a certain decision? Validate and justify why certain key features were responsible in driving certain decisions taken by a model during predictions - ensures ACCOUNTABILITY and RELIABILITY of the model.
* How can we trust model predictions? We should be able to evaluate and validate any data point and how a model takes decisions on it - ensures TRANSPARENCY of the model.
  <br/>
  <br/>
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
* LIME - Local Interpretable Model-Agnostic Explanations
* SHAP

  <br/>
Concept behind global interpretations of model-agnostic feature importance
* We measure a feature’s importance by calculating the increase of the model’s prediction error after perturbing the feature.
* A feature is “important” if perturbing its values increases the model error, because the model relied on the feature for the prediction.
* A feature is “unimportant” if perturbing its values keeps the model error unchanged, because the model basically ignored the feature for the prediction.

### Frameworks
* ELI5
* Skater
* SHAP

### Skater
Frameworks like Skater compute Feature importance based on an information theoretic criteria, measuring the entropy in the change of predictions, given a perturbation of a given feature. The intuition is that the more a model’s decision criteria depend on a feature, the more we’ll see predictions change as a function of perturbing a feature. 

### SHAP (SHapley Additive exPlanations) 
https://www.kdnuggets.com/2018/12/explainable-ai-model-interpretation-strategies.html/2
Frameworks like SHAP, use a combination of feature contributions and game theory to come up with SHAP values. Then, it computes the global feature importance by taking the average of the SHAP value magnitudes across the dataset.  <br/>
Assuming that each feature is a ‘player’ in a game where the prediction is the payout. The Shapley value — a method from coalitional game theory — tells us how to fairly distribute the ‘payout’ among the features. The Shapley value, coined by Shapley, is a method for assigning payouts to players depending on their contribution towards the total payout. Players cooperate in a coalition and obtain a certain gain from that cooperation.
* The ‘game’ is the prediction task for a single instance of the dataset.
* The ‘gain’ is the actual prediction for this instance minus the average prediction of all instances.
* The ‘players’ are the feature values of the instance, which collaborate to receive the gain (= predict a certain value).
The Shapley value is the average marginal contribution of a feature value over all possible coalitions. Coalitions are basically combinations of features which are used to estimate the shapley value of a specific feature. Typically more the features, it starts increasing exponentially hence it may take a lot of time to compute these values for big or wide datasets. 


### Global Surrogate Models
It is a way of building intepretable approximations of really complex models, global surrogate models.  <br/>
A global surrogate model is an interpretable model that is trained to approximate the predictions of a black box model which can essentially be any model regardless of its complexity or training algorithm.  <br/>
Steps involved in building surrogate models:  <br/>
* Choose a dataset This could be the same dataset that was used for training the black box model or a new dataset from the same distribution. You could even choose a subset of the data or a grid of points, depending on your application.
* For the chosen dataset, get the predictions of your base black box model.
* Choose an interpretable surrogate model (linear model, decision tree, …).
* Train the interpretable model on the dataset and its predictions (Predictions from Black box are used as Targets).
* Congratulations! You now have a surrogate model.
* Measure how well the surrogate model replicates the prediction of the black box model.
* Interpret / visualize the surrogate model.

### LIME
LIME explanations are based on local surrogate models. LIME focuses on fitting local surrogate models to explain why single predictions were made instead of trying to fit a global surrogate model.  <br/>
* Choose your instance of interest for which you want to have an explanation of the predictions of your black box model.
* Perturb your dataset and get the black box predictions for these new points.
* Weight the new samples by their proximity to the instance of interest.
* Fit a weighted, interpretable (surrogate) model on the dataset with the variations.
* Explain prediction by interpreting the local model.



