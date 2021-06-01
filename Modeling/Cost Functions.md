## Links
* https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0


## Regression
* SSE - Sum of squared errors
* If a prediction is far from mean, that prediction is penalized more compared to a prediction near to the mean
![](https://miro.medium.com/max/704/1*32P7CzvHv_M2QIG_GjKEJw.jpeg)

## Classification
* Logloss - For Binary
* Cross Entropy - For Multi-class
* Both Logloss & Cross Entropy are essentially the same
* For multi-class: If using Keras, use sigmoid as activation function in last layer and Binary Cross-Entropy (binary_crossentropy) as Loss function. If using tensorflow, use sigmoid_cross_entropy_with_logits

### Logloss
https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a </br>
def log_loss_cond(actual, predict_prob): </br>
  if actual == 1:   </br>
    # use natural logarithm </br>
    return -log(predict_prob)  </br>
  else: </br>
    return -log(1 - predict_prob) </br> (but log(o) is -inf, so we take e^-15)
 </br>
-1/N * Σ ln(prob of the actual target happening). ln is used and not log
* prob of the actual target happening implies 
  * If Target=0 for a observation then what is the probability of 0 happening. 
  * If Target=1 for a observation then what is the probability of 1 happening.
* loss_reg_1 = -1 * np.sum(y_true * np.log(y_pred_1))   +    -1 * np.sum((1 - y_true) * np.log(1 - y_pred_1))
* Which equals to the probability of correctness - for an observation where 0 should happen, if prob of 0 happening is high then logloss i.e -log value is less whereas if prob of 0 happening is less then logloss is more
* Same is the same case for Target=1
* If 'prob of the actual target happening' is high then logloss i.e -log value is less whereas if 'prob of the actual target happening' is less then logloss is more

Probability of it being close to the actual target

![](http://wiki.fast.ai/images/4/43/Log_loss_graph.png)

### Log loss of Class Imbalance data - https://www.coursera.org/learn/ai-for-medical-diagnosis/lecture/qSNmX/impact-of-class-imbalance-on-loss-calculation
*  Log loss = Σ(-log of probability of Y=1 if y=1) + Σ(-log of probability of Y=0 if y=0)
* Modified Log loss = ( (Num Negative/No Total) * Σ(-log of probability of Y=1 if y=1) ) + ( (Num Positive/No Total) * Σ(-log of probability of Y=0 if y=0) )



### Cross-Entropy
Cross-entropy is the more generic form of logarithmic loss when it comes to machine learning algorithms. While log loss is used for binary classification algorithms, cross-entropy serves the same purpose for multiclass classification problems.
* Cross Entropy loss = -1/N * Σ{ylog(p)}. https://stackoverflow.com/questions/41990250/what-is-cross-entropy
* If there are 3 classes - A, B, C.
  * ylog(p) for A = 1*(Prob of A happening)
  * ylog(p) for B = 1*(Prob of B happening)
  * ylog(p) for C = 1*(Prob of C happening)


### loss functions for Regression
##### Mean Squared Error Loss
* if the distribution of the target variable is Gaussian/Normal
* model.compile(loss='mean_squared_error', metrics=['mse']) 

##### Mean Squared Logarithmic Error Loss
* https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error-(msle)
* When target value has a spread of values and when predicting a large value, you may not want to punish a model as heavily as mean squared error. It is kind of Normalized MSE. for skewed data
* first calculates the natural logarithm of each of the predicted values, then calculate the mean squared error
* model.compile(loss='mean_squared_logarithmic_error')

##### Mean Absolute Error Loss
* When distribution of the target variable may be mostly Gaussian, but may have outliers, e.g. large or small values far from the mean value. it is more robust to outliers
* It is calculated as the average of the absolute difference between the actual and predicted values
* MAE is used when outliers are not imp, MSE should be used when outliers are imp
* Drawback of MAE is that gradient is the same throughout, which can be compensated by using dynamic learning rate ( dec learning rate as we mov ecloser to the minima)
* model.compile(loss='mean_absolute_error')


##### Poisson loss function 
* It is used for regression when modeling count data. Use for data follows the poisson distribution. Ex: churn of customers next week. 
* Use the Poisson loss when you believe that the target value comes from a Poisson distribution and want to model the rate parameter conditioned on some input. 
* Examples of this are the number of customers that will enter a store on a given day, the number of emails that will arrive within the next hour, or how many customers that will churn next week.
* Computes the Poisson loss between y_true and y_pred.
* loss = y_pred - y_true * log(y_pred)
* model.compile(loss='mean_absolute_error')

##### Huber loss
* https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
* https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
* Huber loss is less sensitive to outliers in data than the squared error loss. It’s also differentiable at 0. It’s basically absolute error, which becomes quadratic when error is small. How small that error has to be to make it quadratic depends on a hyperparameter, 𝛿 (delta), which can be tuned. Huber loss approaches MSE when 𝛿 ~ 0 and MAE when 𝛿 ~ ∞ (large numbers.)
* The Huber loss combines the best properties of MSE and MAE. It is quadratic for smaller errors and is linear otherwise (and similarly for its gradient). It is identified by its delta parameter

##### logcosh 
* https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
* 'logcosh' works mostly like the mean squared error, but will not be so strongly affected by the occasional wildly incorrect prediction. It has all the advantages of Huber loss, and it’s twice differentiable everywhere, unlike Huber loss.


### loss functions for Classification

##### Binary Cross-Entropy Loss
* It is intended for use with binary classification where the target values are in the set {0, 1}
* model.compile(loss='binary_crossentropy', metrics=['accuracy'])
* Last layer of the model should be
* model.add(Dense(1, activation='sigmoid'))

##### Hinge Loss
* It is intended for use with binary classification where the target values are in the set {-1, 1}. primarily developed for use with Support Vector Machine (SVM) models.
* The hinge loss function encourages examples to have the correct sign, assigning more error when there is a difference in the sign between the actual and predicted class values.
* model.compile(loss='hinge')
* Last layer of the model should be
* model.add(Dense(1, activation='tanh')) # Activation function is tanh as range is [-1,1]

##### Squared Hinge Loss
* It is intended for use with binary classification where the target values are in the set {-1, 1}. 
* If using a hinge loss does result in better performance on a given binary classification problem, is likely that a squared hinge loss may be appropriate.
* It has the effect of smoothing the surface of the error function and making it numerically easier to work with.
* model.compile(loss='squared_hinge')
* Last layer of the model should be
* model.add(Dense(1, activation='tanh')) # Activation function is tanh as range is [-1,1]


### Multi-Class Classification Loss Functions
##### Last layer of the model should be
* model.add(Dense(n, activation='softmax')) # n (number of nodes) is number of classes. this means that the target variable must be one hot encoded before splitting into train and test and before fitting the model

##### Cross-entropy 
* It is the default loss function to use for multi-class classification problems where each class is assigned a unique integer value
* model.compile(loss='categorical_crossentropy')

##### Sparse Multiclass Cross-Entropy Loss
* Sparse cross-entropy addresses this by performing the same cross-entropy calculation of error, without requiring that the target variable be one hot encoded prior to training.
* For example, predicting words in a vocabulary may have tens or hundreds of thousands of categories, one for each label. This can mean that the target element of each training example may require a one hot encoded vector with tens or hundreds of thousands of zero values, requiring significant memory.
* model.compile(loss='sparse_categorical_crossentropy')

##### Kullback Leibler Divergence Loss
* Kullback Leibler Divergence, or KL Divergence for short, is a measure of how one probability distribution differs from a baseline distribution
* KL divergence loss function is more commonly used when using models that learn to approximate a more complex function than simply multi-class classification, such as in the case of an autoencoder used for learning a dense feature representation under a model that must reconstruct the original input. In this case, KL divergence loss would be preferred.
* model.compile(loss='kullback_leibler_divergence')
