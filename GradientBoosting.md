### Links
Working of GBM with eg - https://www.kdnuggets.com/2018/07/intuitive-ensemble-learning-guide-gradient-boosting.html

GBM starts with a weak model for making predictions. The target of such a model is the desired output of the problem. After training such model, its residuals are calculated. If the residuals are not equal to zero, then another weak model is created to fix the weakness of the previous one. But the target of such new model will not be the desired outputs but the residuals of the previous model. That is if the desired output for a given sample is T, then the target of the first model is T. After training it, there might be a residual of R for such sample. The new model to be created will have its target set to R, not T. This is because the new model fills the gaps of the previous models. In summary, 
• We first model data with simple models and analyze data for errors. 
• These errors signify data points that are difficult to fit by a simple model. 
• Then for later models, we particularly focus on those hard to fit data to get them right. 
• In the end, we combine all the predictors by giving some weights to each predictor.

* GBM is used for both regression and classification
* Typically uses decision trees
* Intution behind GBM is to repetitively leverage the patterns in residuals and strengthen a model with weak predictions and make it better

### Parameter Tuning
https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

##### Learning rate/Shrinkage 
* Controls the magnitude of change in estimates after each tree. 
* Lower values are generally preferred as they make the model robust to the specific characteristics of tree and thus allowing it to generalize well but Lower values would require higher number of trees to model all the relations and will be computationally expensive.
* shrinkage is used for reducing, or shrinking, the impact of each additional fitted base-learner (tree). It reduces the size of incremental steps and thus penalizes the importance of each consecutive iteration. The intuition behind this technique is that it is better to improve a model by taking many small steps than by taking fewer large steps. If one of the boosting iterations turns out to be erroneous, its negative impact can be easily corrected in subsequent steps.
* High learn rates and especially values close to 1.0 typically result in overfit models with poor performance.  Values much smaller than .01 significantly slow down the learning process and might be reserved for overnight runs.
* Use a small shrinkage (slow learn rate) when growing many trees.
* One typically chooses the shrinkage parameter beforehand and varies the number of iterations (trees) N with respect to the chosen shrinkage. 










