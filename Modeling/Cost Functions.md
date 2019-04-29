### Regression
* SSE - Sum of squared errors
* If a prediction is far from mean, that prediction is penalized more compared to a prediction near to the mean

### Classification
* Logloss - For Binary
* Cross Entropy - For Multi-class
* Both Logloss & Cross ENtropy are essentially the same

##### Formula
-1/N * Σ log(prob of the actual target happening)
* prob of the actual target happening implies 
  * If Target=0 for a observation then what is the probability of 0 happening. 
  * If Target=1 for a observation then what is the probability of 1 happening.
* Which equals to the probability of correctness - for an observation where 0 should happen, if prob of 0 happening is high then logloss i.e -log value is less whereas if prob of 0 happening is less then logloss is more
* Same is the same case for Target=1
* If 'prob of the actual target happening' is high then logloss i.e -log value is less whereas if 'prob of the actual target happening' is less then logloss is more





