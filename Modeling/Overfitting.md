### Links
https://www.kdnuggets.com/2015/08/feldman-avoid-overfitting-holdout-adaptive-data-analysis.html <br/>
https://arxiv.org/pdf/1506.02629.pdf <br/>
https://www.oreilly.com/ideas/3-ideas-to-add-to-your-data-science-toolkit <br/>
https://ai.googleblog.com/2015/08/the-reusable-holdout-preserving.html <br/> 

### Problem
* Adaptive data analysis: Though we use a holdout set to verify our model built on training set, we use holdout set to revise parameters or algorithm, this frequantly leads to over-fitting on holdout set

### Solution
* Regularization
* Differential Privacy - On an intuitive level, differential privacy hides the data of any single individual. We are thus interested in
pairs of datasets S, S0 that differ in a single element, in which case we say S and S0 are adjacent.

### How Threshold works?
In a nutshell, the reusable holdout mechanism is simply this: access the holdout set only through a suitable differentially private algorithm. It is important to note, however, that the user does not need to understand differential privacy to use our method. The user interface of the reusable holdout is the same as that of the widely used classical method.

### Thresholdout - Reusable holdout sets
* The limit of the method is determined by the size of the holdout set - the number of times that the holdout set may be used grows roughly as the square of the number of collected data points in the holdout, as our theory shows.
* Based on Differential privacy - It is a notion of stability requiring that any single sample should not influence the outcome of the analysis significantly.

### Advantages of Thresholdout
* Stability - Modifying a single data point doesn't chnage outcome too much
* Differential privacy - Notion of privacy-preserving data analysis. Differential privacy is a strong form of stability that allows
adaptive/sequential composition of different analyses
What sets differential privacy apart from other stability notions is that it is preserved by adaptive composition. Combining multiple algorithms that each preserve differential privacy yields a new algorithm that also satisfies differential privacy albeit at some quantitative loss in the stability guarantee. This is true even if the output of one algorithm influences the choice of the next. This strong adaptive composition property is what makes differential privacy an excellent stability notion for adaptive data analysis.

