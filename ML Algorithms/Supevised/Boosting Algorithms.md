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
* Models are updated using gradient descent hence called as GBM

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
* Both XGBoost and LigtGBM handles missing values similarly: Model ignores missing values during a split and then allocate them to whichever side it reduces the loss
https://www.riskified.com/resources/article/boosting-comparison/

### LightGBM
https://towardsdatascience.com/what-makes-lightgbm-lightning-fast-a27cf0d9785e </br>
https://stats.stackexchange.com/questions/319710/lightgbm-understanding-why-it-is-fast  </br>
https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/ </br>
* LightGBM aims to reduce complexity of histogram building by Gradient based one side sampling (GOSS) and Exclusive Feature Bundling (EFB) for finding the optimum split points
* LightGBM aims to reduce complexity of histogram building ( O(data * feature) ) by down sampling data and feature using GOSS and EFB. This will bring down the complexity to (O(data2 * bundles)) where data2 < data and bundles << feature.
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
![](https://www.riskified.com/wp-content/uploads/2019/11/inner-image-trees-1-1024x386.png)
https://www.kdnuggets.com/2019/06/clearing-air-around-boosting.html </br>
https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/ </br>
https://towardsdatascience.com/categorical-features-parameters-in-catboost-4ebd1326bee5 </br>
* Ordered Boosting, Oblivious trees and handling of Categorical vars
* Handles Categorical data such as audio, text, image automatically 
* Performance - CatBoost provides state of the art results and it is competitive with any leading machine learning algorithm on the performance front (Prediction time). 
* Handling Categorical features automatically - We can use CatBoost without any explicit pre-processing to convert categories into numbers. CatBoost converts categorical values into numbers using various statistics on combinations of categorical features and combinations of categorical and numerical features
https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/ (in the xample mention in the link, for 2nd rock: CountInclass is calculate the number of earlier rocks that have Target=1; Total count is no of previous rocks before the current observation)
* Robust - It reduces the need for extensive hyper-parameter tuning and lower the chances of overfitting also which leads to more generalized models. Although, CatBoost has multiple parameters to tune and it contains parameters like the number of trees, learning rate, regularization, tree depth, fold size, bagging temperature and others. 
* Easy-to-use - You can use CatBoost from the command line, using an user-friendly API for both Python and R
* In GBM, trees are built iteratively using the same dataset and this process leads to a bit of target leakage which affects the generalization capabilities of the model. To combat this issue we usa a new variant of Gradient Boosting, called Ordered Boosting. It starts out with creating s+1 permutations of the dataset. This permutation is the artificial time that the algorithm takes into account. Let’s call it \sigma_0\; to\; \sigma_s. The permutations \sigma_1\; to\; \sigma_s is used for constructing the tree splits and \sigma_0 is used to choose the leaf values b_j. In the absence of multiple permutations, the training samples with short “history” will have high variance and hence having multiple permutations ease out that defect.
* Both simple_ctr and combinations_ctr are complex parameters that provide regulation of the categorical features encodings types. While simple_ctr is responsible for processing the categorical features initially present in the dataset, combinations_ctr affects the encoding of the new features, that CatBoost creates by combining the existing features. 
* CatBoost too uses a different kind of Decision Tree, called Oblivious Trees. In such trees the same splitting criterion is used across an entire level of the tree. Such trees are balanced and less prone to overfitting. (it is level-wise algo)
* In oblivious trees each leaf index can be encoded as a binary vector with length equal to the depth of the tree. This fact is widely used in CatBoost model evaluator: it first binarizes all float features and all one-hot encoded features, and then uses these binary features to calculate model predictions. This helps in predicting at very fast speed.
* CatBoost also differs from the rest of the flock in another key aspect – the kind of trees that is built in its ensemble. CatBoost, by default, builds Symmetric Trees or Oblivious Trees. These are trees the same features are responsible in splitting learning instances into the left and the right partitions for each level of the tree. This has a two-fold effect in the algorithm –
Catboost uses oblivious decision trees. Before learning, the possible values of each feature are divided into buckets delimited by threshold values, creating feature-split pairs. Example for such pairs are: (age, <5), (age, 5-10), (age, >10) and so on. In each level of an oblivious tree, the feature-split pair that brings to the lowest loss (according to a penalty function) is selected and is used for all the level’s nodes.
  * Regularization: Since we are restricting the tree building process to have only one feature split per level, we are essentially reducing the complexity of the algorithm and thereby regularization.
  * Computational Performance: One of the most time consuming part of any tree-based algorithm is the search for the optimal split at each nodes. But because we are restricting the features split per level to one, we only have to search for a single feature split instead of k splits, where k is the number of nodes in the level. Even during inference these trees make it lightning fast. It was shown to be 8X faster than XGBoost in inference.
* How catboost handles missing values: For Numeric fetures uses min as default imputing. For Categorical columns, missing values are treated as a seperate group https://catboost.ai/docs/concepts/algorithm-missing-values-processing.html

### Hyperparameters
![](https://cdn-images-1.medium.com/max/1000/1*A0b_ahXOrrijazzJengwYw.png)

### Diff between xgb, light and cat
* Splits
  * Catboost uses oblivious decision trees. Before learning, the possible values of each feature are divided into buckets delimited by threshold values, creating feature-split pairs. Example for such pairs are: (age, <5), (age, 5-10), (age, >10) and so on. In each level of an oblivious tree, the feature-split pair that brings to the lowest loss (according to a penalty function) is selected and is used for all the level’s nodes.
  * lightGBM uses gradient-based one-side sampling (GOSS) that selects the split using all the instances with large gradients (i.e., large error) and a random sample of instances with small gradients. In order to keep the same data distribution when computing the information gain, GOSS introduces a constant multiplier for the data instances with small gradients. Thus, GOSS achieves a good balance between increasing speed by reducing the number of data instances and keeping the accuracy for learned decision trees.
  * XGboost offers several methods for selecting the best split. For example, a histogram-based algorithm that buckets continuous features into discrete bins and uses these bins to find the split value in each node. This method is faster than the exact greedy algorithm, which linearly enumerates all the possible splits for continuous features, but it is slower compared to GOSS that is used by LightGBM.

Tree growth examples
  * Leaf growth
  * Catboost grows a balanced tree.
  * LightGBM uses leaf-wise (best-first) tree growth. It chooses to grow the leaf that minimizes the loss, allowing a growth of an imbalanced tree. Because it doesn’t grow level-wise, but leaf-wise, overfitting can  happen when data is small. In these cases, it is important to control the tree depth. 
  * XGboost splits up to the specified max_depth hyperparameter and then starts pruning the tree backwards and removes splits beyond which there is no positive gain. It uses this approach since sometimes a split of no loss reduction may be followed by a split with loss reduction. XGBoost can also perform leaf-wise tree growth (as LightGBM).

Missing values handling
  * Catboost has two modes for processing missing values, “Min” and “Max”. In “Min”, missing values are processed as the minimum value for a feature (they are given a value that is less than all existing values). This way, it is guaranteed that a split that separates missing values from all other values is considered when selecting splits. “Max” works exactly the same as “Min”, only with maximum values.
  * In LightGBM and XGBoost missing values will be allocated to the side that reduces the loss in each split. 

Feature importance methods
  * Catboost has two methods: The first is “PredictionValuesChange”. For each feature, PredictionValuesChange shows how much, on average, the prediction changes if the feature value changes. A feature would have a greater importance when a change in the feature value causes a big change in the predicted value. This is the default feature importance calculation method for non-ranking metrics. The second method is “LossFunctionChange”. This type of feature importance can be used for any model, but is particularly useful for ranking models. For each feature the value represents the difference between the loss value of the model with this feature and without it. Since it is computationally expensive to retrain the model without one of the features, this model is built approximately using the original model with this feature removed from all the trees in the ensemble. The calculation of this feature importance requires a dataset.
  * LightGBM and XGBoost have two similar methods: The first is “Gain” which is the improvement in accuracy (or total gain) brought by a feature to the branches it is on. The second method has a different name in each package: “split” (LightGBM) and “Frequency”/”Weight” (XGBoost). This method calculates the relative number of times a particular feature occurs in all splits of the model’s trees. This method can be biased by categorical features with a large number of categories. 
  * XGBoost has one more method, “Coverage”, which is the relative number of observations related to a feature. For each feature, we count the number of observations used to decide the leaf node for. 

Categorical features handling
  * Catboost uses a combination of one-hot encoding and an advanced mean encoding. For features with low number of categories, it uses one-hot encoding. The maximum number of categories for one-hot encoding can be controlled by the one_hot_max_size parameter. For the remaining categorical columns, CatBoost uses an efficient method of encoding, which is similar to mean encoding but with an additional mechanism aimed at reducing overfitting. Using CatBoost’s categorical encoding comes with a downside of a slower model. We won’t go into how exactly their encoding works, so for more details see CatBoost’s documentation.
  * LightGBM splits categorical features by partitioning their categories into 2 subsets. The basic idea is to sort the categories according to the training objective at each split. From our experience, this method does not necessarily improve the LightGBM model. It has comparable (and sometimes worse) performance than other methods (for example, target or label encoding).
  * XGBoost doesn’t have an inbuilt method for categorical features. Encoding (one-hot, target encoding, etc.) should be performed by the user.

Who is the winner?
  * We used 4 datasets in our experiment related to fraud prevention in the e-commerce world, with a binary target variable indicating whether an order made by a customer was fraudulent or legitimate. These example training datasets have around 300 features, a relatively high portion (~⅓) of categorical features and roughly 100K observations.
  * The metric we use is weighted AUC (WAUC), which is similar to AUC but allows the use of different weights to different classes. In our real world data, we have orders that were declined and we can only partially tag them. Therefore, we gave these orders a lower weight in the AUC calculation. Our adjusted metric will be further presented in a future blog. 
  * In all experiments we trained on CPU, using an AWS c4.xlarge instance. 

Training speed
  * We compared the training speed, attempting to create as similar as possible conditions to all algorithms. To do this, we trained a model with 4000 trees and a depth of 1 (root node only), with a learning rate of 0.01. 
  * The plot below shows the results of the experiment. Each point represents the average of 3 repeated trainings.
  * Boosting comparison training times -- GBM is several times slower than the other algorithms
  * It is clear that LightGBM is the fastest out of all the other algorithms. CatBoost and XGBoost also present a meaningful improvement in comparison to GBM, but they are still behind LightGBM. 

For our datasets, LightGBM, CatBoost and XGBoost were ~15x, 5x and 3x faster than GBM, respectively.

Accuracy comparison
  * We compared the algorithms’ WAUC on a test set after performing cross-validation hyperparameter tuning across the relevant algorithm’s parameters. We won’t get into the details regarding the chosen hyperparameters, but one thing worth mentioning is that the hyperparameters that were chosen for CatBoost resulted in a more complex model compared to the other algorithms, meaning that the final model almost always had a larger number of trees and a higher depth. 
  * The table below shows the test results of the cross-validation experiment. Each cell represents the average WAUC of 5 repeated experiments.
  * GBM, CatBoost, LightGBM, XGBoost Baseline - 0.9455, 0.9534, 0.9453, 0.9438 Fashion - 0.9826, 0.9833, 0.9815, 0.9814 Retail - 0.9614, 0.9617, 0.96, 0.96 Digital goods - 0.8739, 0.8767, 0.8732, 0.8751
  * In the baseline dataset, CatBoost outperforms the rest by 0.8-1%, which is a meaningful difference. In the other datasets, the differences are not as significant, suggesting that accuracy might not be the main criterion when we choose an algorithm for these datasets. So, it’s important to take into account other criteria such as speed and technical limitations.

