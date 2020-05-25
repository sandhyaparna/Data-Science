Machine learning problem
* what is being predicted, what data is needed. 
* What are some key actions to collect, analyze, predict, and react to the data or predictions? Remembering that different input features might require different kinds of actions. what data are ee analyzing, what data are we predicting? what data are we reacting to?
* What is the API for the problem during production/prediction? Who will use the service? How were they doing it today? 
* Best use case would be to start with a problem for which you are doing manual data analysis today - Think about, what are some of the benefits to replacing parts of that application with machine learning? What kinds of data would you collect if you wanted to do this? Are you collecting the data today? If not, why not?

Things to take care of: <br/>
* Avoid TRAINING SERVING SKEW - Data that is used in batch process should be same as the data stream - Use same code that was used to process historical data during training and reuse it during predictions.
* Performance metric during Training - Scaling to a lot of data
* Performance metric during Prediction - Speed of response
* Magic of ML comes with quantity and not complexity
* simplify user input in APIs

ML Effort Allocation
* Defining KPIs
* Collecting Data - More time
* Building infrastructure - More time
* Optimizing ML Algo - Not as much time as assumed
* Integration 

Path to ML:
* Individual Contributor
* Delegation - Adding more people
* Digitization - Automation
* Big Data
* ML
 <br/>
Difference between ML and Statistics - In ML the idea is that you build the separate model for situation where you have and you dont have data and Hence MISSING values are not imputed. For OUTLIERS, you find enough outliers that becomes something that you can actually train with <br/>

https://www.ritchieng.com/machine-learning-decision-trees/# <br/>
https://topepo.github.io/caret/available-models.html <br/>
http://www.galitshmueli.com/student-projects <br/>
Data Pre-processing http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html <br/>


![](https://4.bp.blogspot.com/-LYwmoJeMiQ0/W3s7iRNv3BI/AAAAAAAAMtk/Y96yOi4QXpAJRci_1Vz4yRlmGiWNzazZQCLcBGAs/s1600/84b03b9bbcb9c5e680e522c35cee6930.png)
<br/>
![](https://media.licdn.com/dms/document/C4E1FAQFzaPKiGHthIw/feedshare-document-pdf-analyzed/0?e=1551074400&v=beta&t=Ia3lVG4RRkp11ywtrclpy6a4CiEeXOYrZdTFBZsdEZE)


https://medium.com/m/callback/email?token=80dc0302a181&operation=login&source=email-eaf21197fd39-1590444931391-auth.login------0-------------------336acd50_b2d1_4dca_8fa4_ed38e3a59e9d



