
## Regression
* SSE - Sum of squared errors
* If a prediction is far from mean, that prediction is penalized more compared to a prediction near to the mean
![](https://miro.medium.com/max/704/1*1alssyRLNBz7CsPWUUAY_g.jpeg)

## Classification
* Logloss - For Binary
* Cross Entropy - For Multi-class
* Both Logloss & Cross Entropy are essentially the same

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


