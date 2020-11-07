https://healthedge.udemy.com/course/recommender-systems/learn/lecture/11700392#overview
https://medium.com/datadriveninvestor/how-to-built-a-recommender-system-rs-616c988d64b2


### Collaborative Filtering
* Looks for people who bought similar products
* Personalized Score - Depends on user
* The matrix must be sparse in order to actually have items to recommend If data is not sparse it implies that eveeryy user has rated every other product and there chance to recommend
* User-User Collaborative filtering: We want to recommend User1 some products - Identify users that have the same products that User1 bought but also who bought xtra products than User1. Look at the similarity of ratings between User1 and other users identified. ANd then recommend using weighted ratings i.e if User1 & User3 are more similar, User2 will have more weighting 
![](https://miro.medium.com/max/963/1*aSq9viZGEYiWwL9uJ3Recw.png)
![](https://miro.medium.com/max/1375/1*YGlwilDLSG10HWf3u28ErQ.png)

### Content based filtering
* Based on attributes of User: Gender, Age, Location etc
* Based on Attributes of Items: Genre of Movie, Year of Release, Lead Actor, Director, Box Office Collection, Budget

### User-User Similarity
![](https://miro.medium.com/max/1375/1*_J9jSJf83J3ohpkWHDRLdQ.png)

### Item-Item Similarity
![](https://miro.medium.com/max/963/1*cnz5qr3Y5xtTQgLxuc6_Wg.png)

### Evaluation Metrics
https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832 </br>
* Rank-Aware Top-N metrics
  * MRR - Mean Reciprocal Rank - It tries to measure “Where is the first relevant item?”. It is closely linked to the binary relevance family of metrics. It puts focus on the first relevant element of the list The algorithm goes as follows: 
    * Generate list of recommendations
    * Find rank ku of its first relevant recommendation for each user: if the first recomended product is correct then 1, if the first recommended product is wrong and second recommended product is correct then 2, if the first&Second recommended products ae wrong but third recommended product is correct then 3 and so on
    * Compute reciprocal rank i.e 1/ku
    * Then average reciprocal rank for the users evaluated
    ![](https://miro.medium.com/max/884/1*dR24Drmb9J5BLZp8ffjOGA.png)
  * MAP - Mean AVerage Precision
    * For each relevant item - compute precision of list through that item
    * Average sub-list precision
    ![](https://miro.medium.com/max/963/1*0xdZ-NWJLlf3m4oyjh0K5g.png)
  * NDCG - Normalized Discounted Cumulative Gain
    * It is able to use the fact that some documents are “more” relevant than others. Highly relevant items should come before medium relevant items, which should come before non-relevant items.
    *
    ![](https://miro.medium.com/max/963/1*W6cQB2kozFxedqVu9lpSVw.png)
* Prediction Accuracy Metrics:
  * MAE : Absolute (Prediction - Rating)
  * MSE : Squares the error - Gives more penalization if difference between prediction-Rating is more
  * RMSE : Square root of MSE. And it is in similar scale to MSE 
* Decision Support - They help the user to select “good” items, and to avoid “bad” items. but they are not targeted to the Top-N recommendations
  * F1 score - Precision & Recall

  








### Links
https://www.ima.umn.edu/2018-2019/DSS8.10.18-5.27.19/27827

Algorithms used in Recommender Systems - Bayesian,Decision Tree,Matrix factorization-based,Neighbor-based ,Neural Network, Rule Learning,Ensemble ,Gradient descent-based,Kernel methods,Clustering ,Associative classification,Bandit, Lazy learning,Regularization methods,Topic Independent Scoring Algorithm <br/>

Collaborative filtering: The process of filtering used by most of the recommender systems to find patterns or information by collaborating viewpoints, various data sources and multiple agents. <br/>

A Recommendation system is a sub-class of information filtering that seeks to predict the rating or preference a user would give to an item. Job of a recommendation algorithm is not to be very predictive of past held-out interactions but is to be predictive of future interactions as users change over time and items chnage over time.
Eg: Amazon-Next to buy, Travel websites-Next Destination, Netflix-Movies, News
* What is recommended
* How is it recommended
* When is it recommended
* Where is it recommended

Matrix foundation 
* AB Test: Consider analysing other factors/confounding vars that might impact the results </br>
How do you measure the outcome - wch outcome is imp? 
  * Customer spensing more time on netflix?
  * Public joy?
  * Long-term rewards
  * Production bias, recommendation bias
  
* Personalization: 
  * Maximize member's enjoyment of the selected show
  * Minimize the time it takes to find them
  * Ordering the titles in each row (Diff categories/genres) is personalized
  * Selection and placement of each row types is personalized
  * Personalized images on same movie
  * Model using
    * User's taste
    * Context: Time, Device, country, Language
    * Difference in local tastes: Not available!= Not popular 

* Latent Models for recommendation:
  * Shallow
    * Latent factor models -- Matrix Factorization
    * Latent Dirichlet Allocation (LDA)
  * Deep
    * Variational Autoencoder
    * Feedforward Neural Networks
    * Sequential Neural Networks (RNN)
    * Convolutional Neural Networks
    
    
    
    


