### Links
Course1 https://www.analyticsvidhya.com/blog/2018/10/introduction-neural-networks-deep-learning/ <br/>
Course2  <br/>
Course3  <br/>
Course4  <br/>
Course5  <br/>

### Computational Time - Activation Function
* A neuron/node/Unit will take an input, apply some activation function (non-linear) to it, and generate an output.
* The purpose of the activation function is to introduce non-linearity into the output of a neuron. This is important because most real world data is non linear and we want neurons to learn these non linear representations.
* Activation function plays an important role in computational time.
* Some of the activation functions:
  * Logistic Sigmoid - σ(x) = 1 / (1 + exp(−x))
  * Hyperbolic Tangent - tanh(x) = 2σ(2x) − 1
  * Rectified Linear Unit - ReLU - f(x) = max(0, x)
  * Softplus
![](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-08-at-11-53-41-am.png) <br/>
Slope, or the gradient of Sigmoid function, at the extreme ends is close to zero. Therefore, the parameters are updated very slowly, resulting in very slow learning. Hence, switching from a sigmoid activation function to ReLU (Rectified Linear Unit) is one of the biggest breakthroughs we have seen in neural networks. ReLU updates the parameters much faster as the slope is 1 when x>0. This is the primary reason for faster computation of the models.
* Sigmoid takes a real-valued input and squashes it to range between 0 and 1
* tanh takes a real-valued input and squashes it to the range [-1, 1]
* ReLU takes a real-valued input and thresholds it at zero (replaces negative values with zero)

##### Why do we need non-linear activation functions?
Using linear activation is essentially pointless. The composition of two linear functions is itself a linear function, and unless we use some non-linear activations, we are not computing more interesting functions. That’s why most experts stick to using non-linear activation functions.


### Performace
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Screenshot-from-2018-10-12-14-29-37-850x438.png) <br/>
As the amount of data increases, the performance of traditional learning algorithms, like SVM and logistic regression, does not improve by a whole lot. In fact, it tends to plateau after a certain point. In the case of neural networks, the performance of the model increases with an increase in the data you feed to the model. <br/>
There are basically three scales that drive a typical deep learning process: <br/>
1. Data <br/>
2. Computation Time <br/>
3. Algorithms <br/>








Backpropagation



