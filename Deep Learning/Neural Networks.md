### Links
Course1 https://www.analyticsvidhya.com/blog/2018/10/introduction-neural-networks-deep-learning/ <br/>
Course2  <br/>
Course3  <br/>
Course4  <br/>
Course5  <br/>
Intro to NN https://www.kdnuggets.com/2016/11/quick-introduction-neural-networks.html

### Computational Time - Activation Function
* A neuron/node/Unit will take an input, apply some activation function (non-linear) to it, and generate an output.
* The purpose of the activation function is to introduce non-linearity into the output of a neuron. This is important because most real world data is non linear and we want neurons to learn these non linear representations.
* Activation function plays an important role in computational time.
* Every activation function (or non-linearity) takes a single number and performs a certain fixed mathematical operation on it  Some of the activation functions:
  * Logistic Sigmoid - σ(x) = 1 / (1 + exp(−x))
  * Hyperbolic Tangent - tanh(x) = 2σ(2x) − 1
  * Rectified Linear Unit - ReLU - f(x) = max(0, x)
  * Softplus
![](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-08-at-11-53-41-am.png) <br/>
Slope, or the gradient of Sigmoid function, at the extreme ends is close to zero. Therefore, the parameters are updated very slowly, resulting in very slow learning. Hence, switching from a sigmoid activation function to ReLU (Rectified Linear Unit) is one of the biggest breakthroughs we have seen in neural networks. ReLU updates the parameters much faster as the slope is 1 when x>0. This is the primary reason for faster computation of the models.
* Sigmoid takes a real-valued input and squashes it to range between 0 and 1
* tanh takes a real-valued input and squashes it to the range [-1, 1]
* ReLU takes a real-valued input and thresholds it at zero (replaces negative values with zero implies slope is 0 when x<0)

##### Why do we need non-linear activation functions?
Using linear activation is essentially pointless. The composition of two linear functions is itself a linear function, and unless we use some non-linear activations, we are not computing more interesting functions. That’s why most experts stick to using non-linear activation functions.

### Performace
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Screenshot-from-2018-10-12-14-29-37-850x438.png) <br/>
As the amount of data increases, the performance of traditional learning algorithms, like SVM and logistic regression, does not improve by a whole lot. In fact, it tends to plateau after a certain point. In the case of neural networks, the performance of the model increases with an increase in the data you feed to the model. <br/>
There are basically three scales that drive a typical deep learning process: <br/>
1. Data <br/>
2. Computation Time <br/>
3. Algorithms <br/>

### Algorithms

#### Back-Propagation Algorithm - When there are more hidden layers
Abbreviated as BackProp. Initially all the edge weights are randomly assigned. For every input in the training dataset, the ANN is activated and its output is observed. This output is compared with the desired output that we already know, and the error is "propagated" back to the previous layer. This error is noted and the weights are "adjusted" accordingly. Weights are adjusted using Gradient Descent Optimization. This process is repeated until the output error is below a predetermined threshold.
* Weights are initialized randomly using the following code = {np.random.randn(No of Input Vars, Number of neurons in the first hidden layer, Number of neurons in the second hidden layer, etc) * 0.01} randn generates random floats from a univariate Normal Distribution of mean 0 & Variance 1. Random values are multiplied with 0.01 to initialize small weights. If we initialize large weights, the activation will be large, resulting in zero slope (in case of sigmoid and tanh activation function). Hence, learning will be slow. So we generally initialize small weights randomly.

### Hyperparameters
* Weights and Bias are parameters
The major difference between parameters and hyperparameters is that parameters are learned by the model during the training time, while hyperparameters can be changed before training the model.
* Learning rate – ⍺
* Number of iterations
* Number of hidden layers
* Units in each hidden layer
* Choice of activation function
* Dropout - Regularization technique to avoid overfitting
#### Hidden layers & Units
* Many hidden units within a layer with regularization techniques can increase accuracy. Smaller number of units may cause underfitting.
#### Dropout
Generally, use a small dropout value of 20%-50% of neurons with 20% providing a good starting point. A probability too low has minimal effect and a value too high results in under-learning by the network.
* Use a larger network. You are likely to get better performance when dropout is used on a larger network, giving the model more of an opportunity to learn independent representations.







Backpropagation



