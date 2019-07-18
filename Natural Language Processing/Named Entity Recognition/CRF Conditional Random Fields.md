https://www.analyticsvidhya.com/blog/2018/08/nlp-guide-conditional-random-fields-text-classification/ </br>
https://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/ </br>
Build CRF Model https://cran.r-project.org/web/packages/crfsuite/vignettes/crfsuite-nlp.html </br>


* CRF is a sequence modelling algorithm. This not only assumes that features are dependent on each other, but also considers the future observations while learning a pattern
  * Hidden Markov Model & MaxEnt Markov Model are also sequence modelling algorithms
  * Hiddem Markov Model considers the future observations around the entities for learning a pattern, but it assumes that the features are independent of each other
  * MaxEnt Markov Model assumes that features are dependent on each other, but does not consider future observations for learning the pattern
* Similar to POS (Part of Speech) tagging
* Annotations are required for building CRF modules





























