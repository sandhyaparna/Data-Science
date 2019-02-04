Data Pre-processing http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html 

* The main side-effect of creating a Target variable using data is you introduce bias in the model, for eg-Sepsis def has one of its condition as SBP<90, so using SBP fetaures within feature engineering introduces bias like - if SBP>90 the SEPSIS is present bacause for Sepsis cases if at all SBP<90 that time is taken as their TOP and hence bias is introduced bacause of creating our Target based on conditions
* In Healthcare - built models for data at differnt points - Reason is that a patient have acquired infection earlier and starts showing syptoms much before the actual test indicators
  * At onset
  * 6 hrs before onset
  * 12 hrs before onset
  
* Can we use Stacking algorithm - 1 model for different subset of variables and combine all the diff models
  * 1 model for time related vars
  * 1 model for demographics related vars
  * 1 model for rate variables - that change with time etc

* Reusuable Holdouts to prevent Over-fitting


