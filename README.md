Machine learning problem
* what is being predicted, what data is needed. 
* What are some key actions to collect, analyze, predict, and react to the data or predictions? Remembering that different input features might require different kinds of actions. what data are ee analyzing, what data are we predicting? what data are we reacting to?
* What is the API for the problem during production/prediction? Who will use the service? How were they doing it today? 

Things to take care of: <br/>
* Avoid TRAINING SERVING SKEW - Data that is used in batch process should be same as the data stream - Use same code that was used to process historical data during training and reuse it during predictions.
* Performance metric during Training - Scaling to a lot of data
* Performance metric during Prediction - Speed of response
* Magic of ML comes with quantity and not complexity
* simplify user input in APIs


https://topepo.github.io/caret/available-models.html <br/>
http://www.galitshmueli.com/student-projects <br/>
Data Pre-processing http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html <br/>

![](https://4.bp.blogspot.com/-LYwmoJeMiQ0/W3s7iRNv3BI/AAAAAAAAMtk/Y96yOi4QXpAJRci_1Vz4yRlmGiWNzazZQCLcBGAs/s1600/84b03b9bbcb9c5e680e522c35cee6930.png)
<br/>
 








