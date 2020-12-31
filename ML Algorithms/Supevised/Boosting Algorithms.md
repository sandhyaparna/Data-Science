* AdaBoost & GBM algorithms differ on how they CREATE THE WEAK/BASE LEARNERS during the iterative process.
* XGBoost and GBM differ in modeling details. XGBoost used a more regularized model formalization to control over-fitting, which gives it better performance.


### AdaBoost - Adaptive Boosting
* AdaBoost is a boosting done on Decision stump(by default but other base classifiers can be used too). Decision stump is a unit depth tree which decides just 1 most significant cut on features
* At each iteration, adaptive boosting changes the sample distribution by modifying the weights attached to each of the instances. It increases the weights of the wrongly predicted instances and decreases the ones of the correctly predicted instances. The weak learner thus focuses more on the difficult instances. After being trained, the weak learner is added to the strong one according to his performance (so-called alpha weight). The higher it performs, the more it contributes to the strong learner
* Good generalization- Suited for any kind of classification problem & Not prone to overfitting
* Sensitive to noisy data and highly affected by outliers because it tries to fit each point perfectly. 

### GBM 
* Doesn’t modify the sample distribution at each iteration for instances. Instead of training on a newly sample distribution, the weak learner trains on the remaining errors (so-called pseudo-residuals) of the strong learner. It is another way to give more importance to the difficult instances. At each iteration, the pseudo-residuals are computed and a weak learner is fitted to these pseudo-residuals. Then, the contribution of the weak learner (so-called multiplier) to the strong one isn’t computed according to his performance on the newly distribution sample but using a gradient descent optimization process. The computed contribution is the one minimizing the overall error of the strong learner
* We try to optimize a loss function
* Gradient Boosted trees are harder to fit than random forests
* Gradient Boosting Algorithms generally have 3 parameters which can be fine-tuned, Shrinkage parameter, depth of the tree, the number of trees. Proper training of each of these parameters is needed for a good fit. If parameters are not tuned correctly it may result in over-fitting.
* As more trees are added, predictions usually improve. But maximum number of trees should be limited by using validation set for early stopping, because more number of trees might overfit the training data
* Standard GBM implementation has no regularization like XGBoost
* A GBM would stop splitting a node when it encounters a negative loss in the split. Thus it is more of a greedy algorithm

### XGBoost - Xtreme Gradient Boostimg
Training Loss + Regularization
* XGBoost uses pre-sorted algorithm & Histogram-based algorithm for computing the best split. Both LightGBM and xgboost utilise histogram based split finding
* Regularization - Helps to reduce overfitting. XGBoost is also known as ‘regularized boosting‘ technique
* Parallel Processing - XGBoost implements parallel processing and is blazingly faster as compared to GBM. Xgboost doesn't run multiple trees in parallel as GBM is sequential/Additive process and each tree can be built only after the previous one i.e you need predictions after each tree to update gradients. Rather it does the parallelization WITHIN a single tree (during its construction) by using openMP to create branches independently
http://zhanpengfang.github.io/418home.html
* Handling Missing Values - XGBoost has an in-built routine to handle missing values
* High Flexibility - XGBoost allow users to define custom optimization objectives and evaluation criteria
* Tree Pruning - XGBoost splits upto the max_depth specified and then start pruning the tree backwards and remove splits beyond which there is no positive gain. whereas GBM would stop splitting a node when it encounters a negative loss in the split
* Built-in Cross-Validation - XGBoost allows user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run. This is unlike GBM where we have to run a grid-search and only a limited values can be tested
* But give lots and lots of data even XGBoost takes long time to train

### Diff between XGBoost, LightGBM, catboot using example and code
https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db </br>
https://bangdasun.github.io/2019/03/21/38-practical-comparison-xgboost-lightgbm/  </br>
* LightGBM decides on splits leaf-wise, i.e., it splits the leaf node that maximizes the information gain, even when this leads to unbalanced trees. In contrast, XGBoost and CatBoost expand all nodes depth-wise and first split all nodes at a given depth before adding more levels. The two approaches expand nodes in a different order and will produce different results except for complete trees. </br>
* Structural Differences in LightGBM & XGBoost: LightGBM uses a novel technique of Gradient-based One-Side Sampling (GOSS) to filter out the data instances for finding a split value while XGBoost uses pre-sorted algorithm & Histogram-based algorithm for computing the best split. Here instances are observations/samples. https://towardsdatascience.com/lightgbm-vs-xgboost-which-algorithm-win-the-race-1ff7dd4917d
* Histogram-based Tree Splitting: In simple terms, Histogram-based algorithm splits all the data points for a feature into discrete bins and uses these bins to find the split value of the histogram. While it is efficient than the pre-sorted algorithm in training speed which enumerates all possible split points on the pre-sorted feature values, it is still behind GOSS in terms of speed. </br>
The amount of time it takes to build a tree is proportional to the number of splits that have to be evaluated. And when you have continuous or categorical features with high cardinality, this time increases drastically. But most of the splits that can be made for a feature only offer miniscule changes in performance. And this concept is why a histogram based method is applied to tree building. The core idea is to group features into set of bins and perform splits based on these bins. This reduces the time complexity from O(#data) to O(#bins).

* 

### LightGBM
https://towardsdatascience.com/what-makes-lightgbm-lightning-fast-a27cf0d9785e </br>
https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/ </br>
* LightGBM aims to reduce complexity of histogram building by Gradient based one side sampling (GOSS) and Exclusive Feature Bundling (EFB) for finding the optimum split points
* Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks
* Since it is based on decision tree algorithms, it splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. Also, it is surprisingly very fast, hence the word ‘Light’
* Leaf wise splits lead to increase in complexity and may lead to overfitting and it can be overcome by specifying another parameter max-depth which specifies the depth to which splitting will occur
* Faster training speed and higher efficiency - Light GBM use histogram based algorithm i.e it buckets continuous feature values into discrete bins which fasten the training procedure
* Lower memory usage - Replaces continuous values to discrete bins which result in lower memory usage
* Better accuracy than any other boosting algorithm - It produces much more complex trees by following leaf wise split approach rather than a level-wise approach which is the main factor in achieving higher accuracy. However, it can sometimes lead to overfitting which can be avoided by setting the max_depth parameter
* Compatibility with Large Datasets - It is capable of performing equally good with large datasets with a significant reduction in training time as compared to XGBOOST
* Parallel learning is supported
* It is not advisable to use LightGBM on small datasets. LightGBM is sensitive to overfitting and can easily overfit small data. Their is no threshold on the number of rows but my experience suggests me to use it only for data with 10,000+ rows
* Gradient-based One-Side Sampling(GOSS): notice that data instances with different gradients play different roles in the computation of information gain. In particular, according to the definition of information gain, those instances with larger gradients(i.e., under-trained instances) will contribute more to the information gain.
Therefore, when down sampling the data instances, in order to retain the accuracy of information gain estimation, should better keep those instances with large gradients (e.g., larger than a pre-defined threshold, or among the top percentiles), and only randomly drop those instances with small gradients.
They prove that such a treatment can lead to a more accurate gain estimation than uniformly random sampling, with the same target sampling rate, especially when the value of information gain has a large range
* Exclusive Feature Bundling Technique for LightGBM:
High-dimensional data are usually very sparse which provides us a possibility of designing a nearly lossless approach to reduce the number of features. Specifically, in a sparse feature space, many features are mutually exclusive, i.e., they never take nonzero values simultaneously. The exclusive features can be safely bundled into a single feature (called an Exclusive Feature Bundle).  Hence, the complexity of histogram building changes from O(#data × #feature) to O(#data × #bundle), while #bundle<<#feature . Hence, the speed for training framework is improved without hurting accuracy.
* What is EFB(Exclusive Feature Bundling)?
  * Remember histogram building takes O(#data * #feature). If we are able to down sample the #feature we will speed up tree learning. LightGBM achieves this by bundling features together. We generally work with high dimensionality data. Such data have many features which are mutually exclusive i.e they never take zero values simultaneously.     * LightGBM safely identifies such features and bundles them into a single feature to reduce the complexity to O(#data * #bundle) where #bundle << #feature.
  * Part 1 of EFB : Identifying features that could be bundled together
  * Intuitive explanation for creating feature bundles
    * Construct a graph with weighted (measure of conflict between features) edges. Conflict is measure of the fraction of exclusive features which have overlapping non zero values.
    * Sort the features by count of non zero instances in descending order.
    * Loop over the ordered list of features and assign the feature to an existing bundle (if conflict < threshold) or create a new bundle (if conflict > threshold).
  * Part 2 of EFB : Algorithm for merging features
    * Calculate the offset to be added to every feature in feature bundle.
    * Iterate over every data instance and feature.
    * Initialise the new bucket as zero for instances where all features are zero.
    * Calculate the new bucket for every non zero instance of a feature by adding respective offset to original bucket of that feature.
* Parameter Tuning. Few important parameters and their usage is listed below : https://www.avanwyk.com/an-overview-of-lightgbm/
  * max_depth : It sets a limit on the depth of tree. The default value is 20. It is effective in controlling over fitting.
  * categorical_feature : It specifies the categorical feature used for training model. 
  * bagging_fraction : It specifies the fraction of data to be considered for each iteration.
  * num_iterations : It specifies the number of iterations to be performed. The default value is 100.
  * num_leaves : It specifies the number of leaves in a tree. It should be smaller than the square of max_depth.
  * max_bin : It specifies the maximum number of bins to bucket the feature values.
  * min_data_in_bin : It specifies minimum amount of data in one bin.
  * task : It specifies the task we wish to perform which is either train or prediction. The default entry is train. Another possible value for this parameter is prediction.
  * feature_fraction : It specifies the fraction of features to be considered in each iteration. The default value is one.
  * scale_pos_weight: the weight can be calculated based on the number of negative and positive examples: sample_pos_weight = number of negative samples / number of positive samples.

### CatBoost - Category Boosting
https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/
* Handles Categorical data such as audio, text, image automatically 
* Performance - CatBoost provides state of the art results and it is competitive with any leading machine learning algorithm on the performance front (Prediction time). 
* Handling Categorical features automatically - We can use CatBoost without any explicit pre-processing to convert categories into numbers. CatBoost converts categorical values into numbers using various statistics on combinations of categorical features and combinations of categorical and numerical features
https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/
* Robust - It reduces the need for extensive hyper-parameter tuning and lower the chances of overfitting also which leads to more generalized models. Although, CatBoost has multiple parameters to tune and it contains parameters like the number of trees, learning rate, regularization, tree depth, fold size, bagging temperature and others. 
* Easy-to-use - You can use CatBoost from the command line, using an user-friendly API for both Python and R

### Hyperparameters
![](https://cdn-images-1.medium.com/max/1000/1*A0b_ahXOrrijazzJengwYw.png)
