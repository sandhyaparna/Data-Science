### Resources
* https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html
* https://stanford-cs329s.github.io/syllabus.html
* https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-model-monitor.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-quality.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-bias-drift.html
    * https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-feature-attribution-drift.html
    * 


### Overview
* Monitoring Machine Learning Models in Production - Data Drift, Model Drift, 
* Find and fix data drift, performance degradation, unexpected bias, or data-integrity issues that hurt business outcomes.
* Monitor Data Quality
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
      * Regression Metrics: mae, mse, rmse, r2
      * Binary Classification Metrics: confusion_matrix, recall, precision, accuracy, recall_best_constant_classifier, precision_best_constant_classifier, accuracy_best_constant_classifier, TPR, TNR, FPR, FNR, receiver_operating_characteristic_curve, precision_recall_curve, auc, f0_5, f1, f2, 
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
 
