## References
http://www.saedsayad.com/decision_tree.htm <br/>
https://www.datasciencecentral.com/profiles/blogs/15-great-articles-about-decision-trees <br/>
https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/ <br/>

scikit learn - http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier  <br/>

Decision tree uses divide & conquer approach, Tree-like graph/Flowchart. The topmost decision node in a tree which corresponds to the best predictor called root node. For nominal attributes, the number of children is usually equal to the number of possible values for the attribute. Hence, it is tested only once. For Numerical attributes, we usually test if the attribute value is greater or less than a determined constant. The attribute may get tested several times for different constants.

## Assumptions
Predictor variables are not independent <br/>
Nonlinear model - Predictor variables have nonlinear relationship among them (There is no equation to express relationship between independent and dependant variables i.e model is constructed based on the observed data <br/>

## Algorithms
ID3 - Information Gain (Binnary splitting of Continuous var) <br/>
C4.5 - Gain ratio (Binnary splitting of Continuous var) <br/>
CART - Gini Index (Binnary splitting of Continuous var) <br/>
CHAID - Chi-Square (Multiway splitting of Continuous var) <br/>

## Splitting Criteria
Splitting is based on the attribute that produces the 'purest' subsets of data w.r.t the label attribute. A partition is pure if all the tuples in it belong to the same class
#### How Decision Tree predicts a Numeric Target?
In case of regression tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mean value
#### How Decision Tree predicts a Categorical Target?
In case of classification tree, the value (class) obtained by terminal node in the training data is the mode of observations falling 
in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mode value.
#### Information Gain:
Info Gain is the difference between entropy of 'Root node' and 'Decision node'. Entropy for each node is the degree 
of homogenity i.e. if a node  is completely homogeneous, its entropy is 0; if a node or var equally divided between say for example 
A,B or C then it is 1. Split happens when Info gain is high. <br/>
ENTROPY -  If the sample is completely homogeneous the entropy is zero and if the sample is equally divided it has entropy of one. <br/>
Split is based on the variable with least entropy <br/>
&nbsp; Entropy using freq table of 1 attribute - i refers to the number of outcomes <br/>
&nbsp; &nbsp; E(S) = Σ - p(i)*log(p(i)) <br/>
&nbsp; Entropy using freq table of 2 attributes  <br/>
&nbsp; &nbsp; E(T,X) = Σ P(c)E(c) <br/>
INFORMATION GAIN (Decrease in Entropy) - The information gain is based on the decrease in entropy after a dataset is split on an attribute.Constructing a decision tree is all about finding attribute that returns the highest information gain (i.e., the most homogeneous branches). <br/>
&nbsp; Gain(T,X) = Entropy(T) - Entropy(T,X)
#### Gain Ratio:
Info gain is biased toward attributes that have a larger number of values(more Unique values) over attributes that have a smaller number of values. Penalizing attributes with large number of values is done using gain ratio.  <br/>
&nbsp; GainRatio(T,X) = Gain(T,X) / SpliInformation(T,X)  <br/>
&nbsp; Split(T,X) = Σ - p(i)*log(p(i))
#### Gini Index:
Variable split is based on the one with low Gini Index. Performs only binary splits
#### Chi-Square:
Finds out the statistical significance between the differences between sub-nodes and parent node. We measure it by sum of squares of standardized differences between observed and expected frequencies of target variable. Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node.  <br/>
&nbsp; Chi-square = ((Actual – Expected)^2 / Expected)^1/2
#### Reduction in Variance:
Used for continuous target variable. Variance formula is used for best split
  
## Binning
Helps reducing the noise or non-linearity. Allows easy identification of outliers, invalid and missing values of numerical variables <br/>
Binning of Continuous variables is performed to avoid binary splitting <br/>
#### Unsupervised
http://www.saedsayad.com/unsupervised_binning.htm <br/>
Entropy  
#### Supervised
http://www.saedsayad.com/supervised_binning.htm <br/>
Do not use target class info during Binning <br/>
size <br/>
Freq <br/>
Rank <br/>
Quantiles <br/>
User-defined

## Overfitting
Overfitting happens when the learning algorithm continues to develop hypotheses that reduce training set error at the cost of an
increased test set error <br/>
#### Pre-pruning
Stops growing the tree earlier, before it perfectly classifies the training set. Tree stops growing when it meets any of these pre-pruning criteria <br/>

MAX DEPTH: Depth of a tree <br/>
MIN SPLIT: Minimum number of records that must exist in a node for a split to happen or be attempted <br/>
MIN LEAFSIZE / MIN BUCKET: Minimum number of records that can be present in a Terminal node <br/>

#### Post-pruning
http://www.saedsayad.com/decision_tree_overfitting.htm <br/>
https://pdfs.semanticscholar.org/025b/8c109c38dc115024e97eb0ede5ea873fffdb.pdf  <br/>
Allows the tree to perfectly classify the training set, and then post prune the tree <br/>

COMPLEXITY PARAMETER: 
Used to control the size of the decision tree and to select the optimal tree size  <br/>
Calculates error complexity for the entire tree before splitting and after splitting. If the difference between errors before splitting and after splitting decreases by atleast cp value mentioned, splitting is useful <br/>
Any split which does not improve the fit by cp will likely be pruned off <br/>
If the cost of adding another variable to the decision tree from the current node is above the value of cp, then tree building does not continue <br/>
cp is similar to min_impurity_decrease in python <br/>

## Disadvantages
Variance is high and if not pruned leads to overfitting/complex tree <br/>
Small variations in data might result in a completely different tree being generated - High Variance. Can be avoided using Bagging or 
Boosting <br/>
It is locally optimized (node by node decisions for splitting) using a greedy algorithm where we cannot guarantee a return to the globally optimal decision tree <br/>
It is an incredibly biased model if a single class takes unless a dataset is balanced before putting it in a tree <br/>

## Advantages
They are incredibly simple to understand due to their visual representation <br/>
They require very little data <br/>
They can handle qualitative and quantitative data <br/>
It can be validated using statistical sets,  <br/>
It can handle large amounts of data  <br/>
It is quite computationally inexpensive <br/>
Performs feature selection  <br/>
Nonlinear relationships between parameters do not affect tree performance <br/>

# Links
https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/ <br/>
Pruning - http://www.saedsayad.com/decision_tree_overfitting.htm <br/>

# General view on DT, how it works, parameter tunuing
# RapidMiner
# Codes in python and R, work on two diff types of datasets - Balanced and UnBalanced, see what all parameters can be changed and how it impacts different datasets

Difference between pruning and pre-pruning?

Splitting happens based on the variable that creates best homogeneous sets


# Stopping Criteria used in splitting - Pruning and Pre-pruning
Pre-pruning that stop growing the tree earlier, before it perfectly classifies the training set.
Post-pruning that allows the tree to perfectly classify the training set, and then post prune the tree. 





