## References
http://www.saedsayad.com/decision_tree.htm
https://www.datasciencecentral.com/profiles/blogs/15-great-articles-about-decision-trees <br/>
https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/ <br/>


## Assumptions
Predictor variables are not independent <br/>
Nonlinear model - Predictor variables have nonlinear relationship among them (There is no equation to express relationship between independent and dependant variables <br/>

## Algorithms
ID3 - Information Gain (Binnary splitting of Continuous var) <br/>
C4.5 - Gain ratio (Binnary splitting of Continuous var) <br/>
CART - Gini Index (Binnary splitting of Continuous var) <br/>
CHAID - (Multiway splitting of Continuous var) <br/>

## Splitting Criteria
#### Information Gain:
Info Gain is the difference between entropy of 'Root node' and 'Decision node'. Entropy for each node is the degree 
of homogenity i.e. if a node  is completely homogeneous, its entropy is 0; if a node or var equally divided between say for example 
A,B or C then it is 1. Split happens when Info gain is high. <br/>
ENTROPY -  If the sample is completely homogeneous the entropy is zero and if the sample is equally divided it has entropy of one. <br/>
Split is based on the variable with least entropy <br/>
&nbsp; Entropy using freq table of 1 attribute - i refers to the number of outcomes <br/>
&nbsp; &nbsp; E(S) = Σ - p(i)*log(p(i)) <br/>
&nbsp; Entropy using freq table of 2 attributes   <br/>
&nbsp; &nbsp; E(T,X) = Σ P(c)E(c) <br/>
INFORMATION GAIN (Decrease in Entropy) - The information gain is based on the decrease in entropy after a dataset is split on an attribute.Constructing a decision tree is all about finding attribute that returns the highest information gain (i.e., the most homogeneous branches). <br/>
&nbsp; Gain(T,X) = Entropy(T) - Entropy(T,X) <br/>
#### Gain Ratio:
Info gain is biased toward attributes that have a larger number of values over attributes that have a smaller number of values. Penalizing attributes with large number of values is done using gain ratio.  <br/>
&nbsp; GainRatio(T,X) = Gain(T,X) / SpliInformation(T,X)  <br/>
&nbsp; Split(T,X) = Σ - p(i)*log(p(i))<br/>  <br/>
#### Gini Index:
Variable split is based on the one with low Gini Index. Performs only binary splits  <br/>
#### Chi-Square:
Finds out the statistical significance between the differences between sub-nodes and parent node. We measure it by sum of squares of standardized differences between observed and expected frequencies of target variable. Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node.  <br/>
&nbsp; Chi-square = ((Actual – Expected)^2 / Expected)^1/2  <br/>
#### Reduction in Variance:
Used for continuous target variable. Variance formula is used for best split

         
## Binning
Binning of Continuous variables is performed to avoid binary splitting <br/>

## DisAdvantages
Small variations in data might result in a completely different tree being generated - High Variance. Can be avoided using Bagging or 
Boosting <br/>


# Links
https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
Pruning - http://www.saedsayad.com/decision_tree_overfitting.htm

# General view on DT, how it works, parameter tunuing
# RapidMiner
# Codes in python and R, work on two diff types of datasets - Balanced and UnBalanced, see what all parameters can be changed and how it impacts different datasets

Difference between pruning and pre-pruning?

Splitting happens based on the variable that creates best homogeneous sets

# How Decision Tree predicts a Numeric Target?
In case of regression tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that
region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mean value

# How Decision Tree predicts a Categorical Target?
In case of classification tree, the value (class) obtained by terminal node in the training data is the mode of observations falling 
in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mode value.

# Stopping Criteria used in splitting - Pruning and Pre-pruning
Pre-pruning that stop growing the tree earlier, before it perfectly classifies the training set.
Post-pruning that allows the tree to perfectly classify the training set, and then post prune the tree. 





