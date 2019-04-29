
### Regression
* SSE - Sum of squared errors
* If a prediction is far from mean, that prediction is penalized more compared to a prediction near to the mean

### Classification
* Logloss - For Binary
* Cross Entropy - For Multi-class
* Both Logloss & Cross ENtropy are essentially the same

##### Logloss
https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a </br>
-1/N * Σ log(prob of the actual target happening)
* prob of the actual target happening implies 
  * If Target=0 for a observation then what is the probability of 0 happening. 
  * If Target=1 for a observation then what is the probability of 1 happening.
* Which equals to the probability of correctness - for an observation where 0 should happen, if prob of 0 happening is high then logloss i.e -log value is less whereas if prob of 0 happening is less then logloss is more
* Same is the same case for Target=1
* If 'prob of the actual target happening' is high then logloss i.e -log value is less whereas if 'prob of the actual target happening' is less then logloss is more

![](http://wiki.fast.ai/images/4/43/Log_loss_graph.png)

##### Cross-Entropy
Cross-entropy is the more generic form of logarithmic loss when it comes to machine learning algorithms. While log loss is used for binary classification algorithms, cross-entropy serves the same purpose for multiclass classification problems.
* Cross Entropy loss = -1/N * Σ{ylog(p)}. https://stackoverflow.com/questions/41990250/what-is-cross-entropy
* If there are 3 classes - A, B, C.
  * ylog(p) for A = 1*(Prob of A happening)
  * ylog(p) for B = 1*(Prob of B happening)
  * ylog(p) for C = 1*(Prob of C happening)


