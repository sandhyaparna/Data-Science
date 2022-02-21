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
* Create a baseline from the dataset that was used to train the model - The baseline computes metrics and suggests constraints for the metrics. Real-time predictions from your model are compared to the constraints, and are reported as violations if they are outside the constrained values.
* Inspect the reports, which compare the latest data with the baseline, and watch for any violations reported and for metrics 
* Constraint suggestion with baseline/training dataset:
  * The training dataset data schema and the inference dataset schema should exactly match i.e. the number and order of the features
  * Name of the feature
  * inferred_type: "Integral" | "Fractional" | "String" | "Unknown" - 
  * 



