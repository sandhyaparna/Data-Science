### Bias-Variance
* Bias is how far model's predictions are from correctness
* Variance is the degree to which these predictions vary between model iterations.
![](https://www.kdnuggets.com/wp-content/uploads/bias-and-variance.jpg)

### Low Bias & Low Variance are desired
![](https://www.kdnuggets.com/wp-content/uploads/bias-variance-total-error.jpg)
* Model is complex = Bias is low and Variance is high
* Model is simple = Bias is high and Variance is low
* High Bias = Both Training and Test error are high
* High Variance = Training error is low but Test error is high

### Model Performance
* High Variance or Overfitting: Train set error = 1%, Validation Set error = 11%
* High Bias or Underfitting: Trainset error = 15%, Validation Set error = 16%
* High Bias and High Variance: Trainset error = 15%, Validation Set error = 30%
* Low Bias and Low Variance: Train set error = 0.5%, Validation Set error = 1%
* 
* If the dev set error is much more than the train set error, the model is overfitting and has a high variance
* When both train and dev set errors are high, the model is underfitting and has a high bias
* If the train set error is high and the dev set error is even worse, the model has both high bias and high variance
* And when both the train and dev set errors are small, the model fits the data reasonably and has low bias and low variance

### ML algos
* The k-nearest neighbours algorithm has low bias and high variance, but the trade-off can be changed by increasing the value of k which increases the number of neighbours that contribute to the prediction and in turn increases the bias of the model.
* The support vector machine algorithm has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.
 <br/>
