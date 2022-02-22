![](http://www.storywarren.com/wp-content/uploads/2016/09/space-1.jpg) 
|:--:| 
| *Space* |

### Resources
* https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html
* https://stanford-cs329s.github.io/syllabus.html
* https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide
* https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-model-monitor.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-quality.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-bias-drift.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-feature-attribution-drift.html
    * 


### Overview
https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide </br>
* Monitoring Machine Learning Models in Production - Data Drift, Model Drift, 
* Find and fix data drift, performance degradation, unexpected bias, or data-integrity issues that hurt business outcomes.
* Essentially, the goal of monitoring your models in production is:
   * To detect problems with your model and the system serving your model in production before they start to generate negative business value,
   * To take action by triaging and troubleshooting models in production or the inputs and systems that enable them,
   * To ensure their predictions and results can be explained and reported, 
   * To ensure the model’s prediction process is transparent to relevant stakeholders for proper governance, 
   * Finally, to provide a path for maintaining and improving the model in production.
* Data Quality issues
   * Testing input data for duplicates,
   * Testing input data for missing values,
   * Catching syntax errors,
   * Catching data type and format errors,
   * Checking schema for semantic errors in terms of feature names,
   * Effective data profiling for complex dependencies in the data pipeline,
   * General integrity checks; does the data meet the requirements of downstream services or consumers?

   * Possible solutions after detecting data quality issues
      * Provide an alert following a schema change.
      * Ensure proper data validation practices are implemented by data owners. 
      * Ensure everyone is aware of their role in getting the data to the pipeline and enable effective communication between data owners so that when a change is made at the data source, the model owner(s) and other service owners are aware.
* Data drift - Data drift refers to a meaningful change in distribution between the training data and production data. Changes in input data distribution will affect model performance over time, although it’s a slower process than in the case of data quality issues.
   * To detect data drift, perform distribution tests by measuring distribution changes using distance metrics:
   * Basic statistical metrics you could use to test drift between historical and current features are; mean/average value, standard deviation, minimum and maximum values comparison, and also correlation. 
   * For continuous features, you can use divergence and distance tests such as Kullback–Leibler divergence, Kolmogorov-Smirnov statistics (widely used),
   * Population Stability Index (PSI), Hellinger distance, and so on.
   * For categorical features, chi-squared test, entropy, the cardinality or frequency of the feature.
   * Some platforms (such as Fiddler) are now providing out-of-the-box monitoring solutions for outlier detection using machine learning and other unsupervised methods.
   * If the features are enormous, as is the case with a lot of datasets, you may want to prune them using dimensionality reduction techniques (such as PCA) and then perform the necessary statistical test.
   * Possible solutions after Data Drift detection
      * The most plausible solution is to trigger an alert and send a notification to the service owner. You might want to use an orchestration tool to kick off a retraining job with production data, and if the distribution change is really large, you might want to build another model with your new data.
      * Oftentimes, your new data won’t be large enough for retraining your model or remodeling. So, you could combine and prepare your new data with historical (training) data and then, during retraining, assign higher weights to the features that drifted significantly from each other.
      * In other cases, you might be lucky to have your new production data sufficient for the task. In such a case, you can go ahead and build a challenger model(s), deploy it (either offline or online), and test using shadow testing or A/B testing approaches to determine if it’s better than or as good as the champion (current) model in production.
* Outliers
   * Use the tests we discussed in the previous section to determine if values and distributions of features are drastically different from normal benchmark periods—very noticeable drifts.
   * Perform statistical distance tests on single events or a small number of recent events detecting out-of-distribution issues.
   * Analyze if the features your model is most sensitive to—the most important features your model learned after training—have changed drastically.
   * Use any of the suitable distribution tests to determine how far off the features (outliers) are from the features in the training set.
   * Use unsupervised learning methods to categorize model inputs and predictions, allowing you to discover cohorts of anomalous examples and predictions. Some platforms use AutoML to detect outliers that your test can’t catch.
   * Possible solutions after outlier detection
      * Perform data slicing methods on sub-datasets to check model performance for specific subclasses of predictions. You can automate this process as your model makes and logs predictions to an evaluation store using your monitoring tool.
      * If your model keeps performing poorly based on your metrics, you might want to consider evaluating the model at its current state and then training a new challenger model. 
      * Document the issue and track if this is a seasonal outlier or an extreme, one-off outlier so you can strategize how to go about troubleshooting such problems in the future.
      * If the performance of the model can’t be improved after retraining or the new model can’t quite cut it, you might want to consider the model’s performance benchmark and perhaps have a human in the loop, assisting the decision process for that period.
* Model drift 
   * Instantaneous model drift: Happens when there’s a sudden drop in model performance over time. It could be a bug in the data pipeline causing data quality issues, or the model being deployed in a new domain, or outlier events (like a global crisis).
   * Gradual model drift: Most common type of model drift happens as a result of the natural consequences of a dynamic, changing, and evolving business landscape. It could happen as a result of user preferences changing over time, new demographics of customers adopting your product, or newly introduced features that skew the underlying pattern in the data.
   * Recurring model drift: This is the result of seasonal events that are periodic and recurring over a year—the pattern is always known and can be forecasted. These could be holidays and yearly discounts. In most cases, user preferences are seasonal or one model serves different regions.
   * Temporary model drift: This is quite difficult to detect by rule-based methods and is often detected using unsupervised methods. It happens due to strange, one-off events such as adversarial attacks, users using the product in a way that was not intended, a model temporarily serving newer clients, or system performance issues.
   * Model drift detection
      * You can detect model/concept drift using the same statistical tests as in the case of data drift.
      * Monitoring predictive performance (with evaluation metrics) of your model is reduced over time. By setting a predictive metrics threshold, you can confirm if your model consistently returns unreliable results and then analyze the prediction drift (changes in prediction results over time) from there. 
      * Monitoring data drift can give you a heads-up on whether you should analyze your model for degradations or drifts.
      * Monitor label drift (changes in the distribution of real labels, for supervised learning solutions) when you can compare ground truth/actual labels to your model’s prediction to analyze trends and new interpretations of data.
   * Possible solutions after detecting model/concept drift
      * Keep monitoring and retraining deployed models according to your business reality. If your business objectives and environment change frequently, you may want to consider automating your system to schedule and execute retraining at predefined intervals compared to more stable businesses (learn more about retraining here).
      * If retraining your models doesn’t improve performance, you may want to consider remodeling or redeveloping models from scratch.
      * If you’re working on larger scale projects with a good budget and little trade-off between cost and performance (in terms of how well your model catches up with a very dynamic business climate), you may want to consider online learning algorithms for your project. 
* Model configuration and artifacts - Track the configurations for relevance—especially the hyperparameter values used by the model during retraining for any abnormality. The model configuration file and artifacts contain all the components that were used to build that model, including:
   * Training dataset location and version,
   * Test dataset location and version,
   * Hyperparameters used,
   * Default feature values,
   * Dependencies and their versions; you want to monitor changes in dependency versions to easily find them for root cause analysis when model failure is caused by dependency changes, 
   * Environment variables,
   * Model type (classification vs regression),
   * Model author,
   * Target variable name,
   * Features to select from the data,
   * Code and data for testing scenarios,
   * Code for the model and its preprocessing.
* Model versions - Monitoring model versions in production are critical if you want to be sure that the right version is deployed.
* Prediction drift - Track model performance metrics by comparing predictions with actual labels (Different metrics are used for classification, regression, clustering, reinforcement learning, and so on)

![](https://cloud.google.com/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd.svg)
*CI/CD Pipeline*


### AWS
* Monitor Data Quality - Data Quality issues, Data drift, outliers
   * Create a baseline from the dataset that was used to train the model - The baseline computes metrics and suggests constraints for the metrics. Real-time predictions from your model are compared to the constraints, and are reported as violations if they are outside the constrained values.
   * Inspect the reports, which compare the latest data with the baseline, and watch for any violations reported and for metrics 
   * Constraint suggestion with baseline/training dataset: https://github.com/awslabs/deequ | https://github.com/awslabs/deequ/blob/master/src/main/scala/com/amazon/deequ/examples/constraint_suggestion_example.md | 
     * The training dataset data schema and the inference dataset schema should exactly match i.e. the number and order of the features
     * Name of the feature
     * inferred_type: "Integral"/integers | "Fractional"/numeric float | "String" | "Unknown"
     * completeness: number, # denotes observed non-null value percentage
     * is_non_negative: boolean
   * Statistics file 
     * name: "feature-name",
     * inferred_type: "Fractional" | "Integral",
     * numerical_statistics: num_present: number, num_missing: number
       * "mean": number
       * "sum": number
       * "std_dev": number
       * "min": number
       * "max": number
     * string type
       * num_missing
       * distinct count
       * each unique string group count
    * **Violation check**
      * data_type_check - If the data types in the current execution are not the same as in the baseline dataset, this violation is flagged. During the baseline step, the generated constraints suggest the inferred data type for each column. The monitoring_config.datatype_check_threshold parameter can be tuned to adjust the threshold on when it is flagged as a violation.
      * completeness_check - If the completeness (% of non-null items) observed in the current execution exceeds the threshold specified in completeness threshold specified per feature, this violation is flagged. During the baseline step, the generated constraints suggest a completeness value.
      * baseline_drift_check - If the calculated distribution distance between the current and the baseline datasets is more than the threshold specified in monitoring_config.comparison_threshold, this violation is flagged.
      * missing_column_check - If the number of columns in the current dataset is less than the number in the baseline dataset, this violation is flagged.
      * extra_column_check - If the number of columns in the current dataset is more than the number in the baseline, this violation is flagged.      
      * categorical_values_check - If there are more unknown values in the current dataset than in the baseline dataset, this violation is flagged. This value is dictated by the threshold in monitoring_config.domain_content_threshold.
* Monitor Model Quality / Model Quality Baseline
   * Model Quality Metrics - https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
      * Regression Metrics: mae, mse, rmse, mape(%), r2, Adj r2
      * Binary Classification Metrics: confusion_matrix, ROC-AUC score,recall, precision, accuracy, recall_best_constant_classifier, precision_best_constant_classifier, accuracy_best_constant_classifier, TPR, TNR, FPR, FNR, receiver_operating_characteristic_curve, precision_recall_curve, auc, f0_5, f1, f2, 
      * Multiclass Metrics
   * Model quality monitoring compares the predictions your model makes with ground truth labels to measure the quality of the model.
   * create an alarm when a specific model quality metric doesn't meet the threshold you specify.
* Bias Drift
   * For example, consider the DPPL bias metric. Specify an allowed range of values A=(amin​,amax​), for instance an interval of (-0.1, 0.1), that DPPL should belong to during deployment. Any deviation from this range should raise a bias detected alert. 
   * For example, you can set the frequency of the checks to 2 days. This means that SageMaker Clarify computes the DPPL metric on data collected during a 2-day window.
   *  Bias is measured by computing a metric and comparing it across groups. The group of interest is specified using the “facet.” For post-training bias, the possitive label should also be taken into account.
*  Feature Drift - https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-feature-attribution-drift.html
   * We can detect the drift by comparing how the ranking of the individual features changed from training data to live data. 
   * Model explainability monitor can explain the predictions of a deployed model producing inferences and detect feature attribution drift on a regular basis.
