Data Pre-processing http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html 

* Generalization is very imp - Can be done using regularization.
* The main side-effect of creating a Target variable using data is you introduce bias in the model, for eg-Sepsis def has one of its condition as SBP<90, so using SBP fetaures within feature engineering introduces bias like - if SBP>90 the SEPSIS is present bacause for Sepsis cases if at all SBP<90 that time is taken as their TOP and hence bias is introduced bacause of creating our Target based on conditions
* Different thresholds can be used for different groups
* Evaluate Right Predictions vs Wrong predictions by groups within a var. For eg - if a var is region, evaluate the perf of predictions within each region
* The key difference between statistics and machine learning, is how we deal with outliers. In statistics, outliers tend to be removed. But in machine learning, outliers tend to be learned. And if you want to learn outliers, you need to have enough examples of those outliers, which essentially means that you have to work with all of your data. You have to have the distribution of outliers, distributions of rare values throughout your dataset. And in order to do that, you have to work with your complete dataset.
* The trick is to do the first part of your aggregation in BigQuery, get back a Pandas DataFrame, then work with the smaller Pandas DataFrame locally.
*
* In Healthcare - built models for data at differnt points - Reason is that a patient have acquired infection earlier and starts showing syptoms much before the actual test indicators
  * At onset
  * 6 hrs before onset
  * 12 hrs before onset
* Try understanding the assumptions and the real-time data flow
  * For a Patient - Admission time is not the exact time of Arrival - Different type of visits like ED, Inpatient, etc have differnt Visit IDs. Modeling should be based on MRN.
    * A patient usually starts as an ED and then gets admitted as InPatient. So, Patient Visits should be based on different type of visit and not within same cohort of InPatients
   * For Sepsis on Arrival - Would temps, labs etc will be taken ??? - What are the filters that should be used forthe model
     * If we assume that patients should have so and so reading taken, blood culture drawn etc conditions, how useful will our model actually be??
   * Duscharge time may be before the last collection data time of a Patient or much beyound the last collection date time of a patient - if there is no vital signs, labs taken for a patient - can we assume that the patient is still in hospital??
   
   
* Can we use Stacking algorithm - 1 model for different subset of variables and combine all the diff models
  * 1 model for time related vars
  * 1 model for demographics related vars
  * 1 model for rate variables - that change with time etc

* Reusuable Holdouts to prevent Over-fitting


#### Personalization:
* What locations/beds are there ina hospital.
* How are coma patients' vital/lab details taken. How do you monitor a coma patient
* In what units is Temp/Vitals signs taken?
* What details do Vigi gets froma client? - If we dont get a particular vital sign values, we cannot include that in our model







