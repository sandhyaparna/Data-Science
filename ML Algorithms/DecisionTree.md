## References
http://www.saedsayad.com/decision_tree.htm <br/>
https://www.datasciencecentral.com/profiles/blogs/15-great-articles-about-decision-trees <br/>
https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/ <br/>

scikit learn - http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier  <br/>

Decision tree uses divide & conquer approach, Tree-like graph/Flowchart. The topmost decision node in a tree which corresponds to the best predictor called root node. For nominal attributes, the number of children is usually equal to the number of possible values for the attribute. Hence, it is tested only once. For Numerical attributes, we usually test if the attribute value is greater or less than a determined constant. The attribute may get tested several times for different constants.
#### How Decision Tree predicts a Numeric Target?
In case of regression tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mean value
#### How Decision Tree predicts a Categorical Target?
In case of classification tree, the value (class) obtained by terminal node in the training data is the mode of observations falling 
in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mode value.

## Assumptions
Predictor variables are not independent <br/>
Nonlinear model - Predictor variables have nonlinear relationship among them (There is no equation to express relationship between independent and dependant variables i.e model is constructed based on the observed data <br/>

## Algorithms
#### ID3 (Iternative Dichotomizer)
Greedy search using Entropy or Information Gain <br/>
It creates good splits at top but dont consider what happens later on the splits <br/>
Doesn't handle numeric attributes and missing values <br/>


#### C4.5 
Gain ratio (Binnary splitting of Continuous var) <br/>
Handles numeric attributes and missing values automatically using surrogate splits


#### C5.0 
Uses less memory and builds smaller rulesets than C4.5 while being more accurate

#### CART 
Twoing criteria (or) Gini Index (Binnary splitting of Continuous var) <br/>
Twoing rules strikes a balance between purity and creating roughly equal-sized nodes [Not available in rpart package] <br/>
Goodness of fit measure: misclassification rates
Regression trees uses - Sum of squared errors <br/>
Goodness of fit measure: Sum of squared errors
Handles numeric attributes and missing values automatically using surrogate splits

#### CHAID 
Chi-Square (Multiway splitting of Continuous var) <br/>


## 
Resursive partitioning - Splitting of population into sub-populations and each sub-population may in turn be split an indefinite number of times until the splitting process terminates after a particular stopping criterion is reached

## Splitting Criteria
Splitting is based on the attribute that produces the 'purest' subsets of data w.r.t the label attribute. A partition is pure if all the tuples in it belong to the same class
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
Gain ratio is ratio of Information gain to the intrinsic information <br/>
&nbsp; GainRatio(T,X) = Gain(T,X) / SpliInformation(T,X)  <br/>
&nbsp; SplitInfo(T,X) = Σ - p(i)*log(p(i))  <br/>
#### Gini Index:
Variable split is based on the one with low Gini Index <br/
Performs only binary splits <br/
Gini produces small but pure nodes <br/>
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
#### Cross-validation
#### Pre-pruning
Allows the tree to perfectly classify the training set, and then post prune the tree <br/> 
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

## Advantages
Incredibly simple to understand & interpret due to their visual representation <br/>
They require very little data preperation, handles misisng and outliers, both qualitative and quantitative variables <br/>
Handles multi-classification problems <br/>
Performs feature selection  <br/>
Model can be validated using statistical sets <br/>
Nonlinear relationships between parameters do not affect tree performance <br/>

## Disadvantages
Variance is high (incase of long depth trees) and if not pruned leads to overfitting/complex tree <br/>
Trees are unstable since small variations in data might result in a completely different tree being generated - Can be avoided using ensemble models like Bagging or Boosting <br/>
It is locally optimized (decisions are made at each node) using a greedy algorithm where we cannot guarantee a return to the globally optimal decision tree - Can be mitigated by training multiple trees in an ensemble learner  <br/>
Create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to using in a tree <br/>





How should the input datasets for Train and Test be?
How to run cross validation within Decision trees? Mention Sampling type for cross-validation?
How to apply model to performance metrics of the decision tree from cv? AUC/Accuracy with 95% CI limits
How to get the confusion matric of the train performance
Decision tree - visulaization?
Decision tree - rules

# Different functions
Label encoder - To convert chars vars
DecisionTreeClassifier
.fit
.predict_proba - Difference between predict & predict_proba
.roc_auc_score
.model_selection.cross_val_score
GridSearchCV

