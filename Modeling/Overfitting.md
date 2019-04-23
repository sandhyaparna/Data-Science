### Links
https://www.kdnuggets.com/2015/08/feldman-avoid-overfitting-holdout-adaptive-data-analysis.html <br/>
https://arxiv.org/pdf/1506.02629.pdf <br/>
https://www.oreilly.com/ideas/3-ideas-to-add-to-your-data-science-toolkit <br/>
https://ai.googleblog.com/2015/08/the-reusable-holdout-preserving.html <br/> 

### Solutions
* Regularization
* Cross-Validation
* Early Stopping - Stopping point should be where Loss on Training set decreases but Validation set increases
* Thresholdout
  * Differential Privacy - On an intuitive level, differential privacy hides the data of any single individual. We are thus interested in pairs of datasets S, S0 that differ in a single element, in which case we say S and S0 are adjacent.
  * Adaptive data analysis: Though we use a holdout set to verify our model built on training set, we use holdout set to revise parameters or algorithm, this frequantly leads to over-fitting on holdout set. Adaptive data analysis is to use a seperate holdout dataset to validate any finding obtained via adaptive analysis.

### Regularization
Complex models are bad. It is the process of adding a tuning parameter to a model to induce smoothness in order to prevent overfitting. One of the ways to keep our model simple is by applying regularization and adjust the rate until we achieve an acceptable performance. 
* It discourages learning a more complex model
* A new term is added to the loss function(SSE in Regression) 
* Ridge = SSE + α(Sum of square of coefficients) - Square of coeffs implies power=2 hence L2
* Lasso = SSE + α(Absolute value of coefficients) - Absolute value implies power=1 hence L1
* Elastic Net = SSE + α2(Sum of square of coefficients) + α1(Absolute value of coefficients)
  * α = 0: Same as Simple linear regression
  * α = ∞: Coeffs will be 0
  * 0 < α < ∞: how much we want to penalize the flexibility of our model.


##### Ways
* Early stopping
* Parameter Norm Penalities
  * L1 regularization
  * L2 regularization
  * Max-norm regularization
* Dataset Augmentation
* Noise robustness
* Sparse representations

##### 


### Thresholdout
##### How Thresholdout works?
Implementation in Python https://github.com/bmcmenamin/thresholdOut-explorations/blob/master/Threshold%20out%20demos%20--%20tuning%20parameters%20for%20linear%20regression.ipynb <br/>
https://andyljones.tumblr.com/post/127547085623/holdout-reuse <br/>
Based on 2 key Ideas:
* First, the validation should not reveal any information about the holdout dataset if the analyst does not overfit to the training set.
* Second, an addition of a small amount of noise to any validation result can prevent the analyst from overfitting to the holdout set. <br/>
In a nutshell, the reusable holdout mechanism is simply this: access the holdout set only through a suitable differentially private algorithm. It is important to note, however, that the user does not need to understand differential privacy to use our method. The user interface of the reusable holdout is the same as that of the widely used classical method.

##### Thresholdout - Reusable holdout sets
* The limit of the method is determined by the size of the holdout set - the number of times that the holdout set may be used grows roughly as the square of the number of collected data points in the holdout, as our theory shows.
* Based on Differential privacy - It is a notion of stability requiring that any single sample should not influence the outcome of the analysis significantly.

##### Advantages of Thresholdout
* Stability - Modifying a single data point doesn't chnage outcome too much
* Differential privacy - Notion of privacy-preserving data analysis. Differential privacy is a strong form of stability that allows
adaptive/sequential composition of different analyses
What sets differential privacy apart from other stability notions is that it is preserved by adaptive composition. Combining multiple algorithms that each preserve differential privacy yields a new algorithm that also satisfies differential privacy albeit at some quantitative loss in the stability guarantee. This is true even if the output of one algorithm influences the choice of the next. This strong adaptive composition property is what makes differential privacy an excellent stability notion for adaptive data analysis.

