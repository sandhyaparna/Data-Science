### Links
https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/ </br>
How Hyperplane is created https://www.coursera.org/learn/launching-machine-learning/lecture/atDbV/short-history-of-ml-kernel-methods

### Overview
* “Support Vector Machine” (SVM), non-linear model, can be used for both classification or regression challenges. 
* It is widely used in Binary classification problems. 
* Classification is performed by choosing a hyper-plane that differentiate the two classes very well. SVM classifiers aim to maximize the margin between the two support vectors using a hinge loss function
* In SVMs, we include two parallel hyperplanes on either side of the decision boundary hyperplane where they intersect with the closest data point on each side of the hyperplane. These are the support vectors. The distance between the two support vectors is the margin.
* Support Vectors are simply the co-ordinates of individual observation where the margin is large. Support Vector Machine is a frontier which best segregates the two classes (hyper-plane/ line). Selecting hyper-plane:
  * Select the hyper-plane which segregates the two classes better
  * After selecting the hyper-plane that segregates 2 class well then choose hyper-plane with higher margin (Margin is distance between nearest data point (either class) and hyper-plane)
* Kernel trick is where low dimensional input space is transformed into a higher dimensional space i.e. it converts not separable problem to separable problem, these functions are called kernels.
  * Linear
  * Polynomial
  * Gaussian Radial Basis 
  
### Adavantages
* It works really well with clear margin of separation
* It is effective in high dimensional spaces.
* It is effective in cases where number of dimensions is greater than the number of samples.
* It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Better for text-classification and Binary

### Disadvantages
* It doesn’t perform well, when we have large data set because the required training time is higher
* It also doesn’t perform very well, when the data set has more noise i.e. target classes are overlapping
* SVM doesn’t directly provide probability estimates, these are calculated using an expensive five-fold cross-validation. It is related SVC method of Python scikit-learn library.







