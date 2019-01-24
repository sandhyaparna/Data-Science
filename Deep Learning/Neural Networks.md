### Links
Course1 https://www.analyticsvidhya.com/blog/2018/10/introduction-neural-networks-deep-learning/
Course2 
Course3 
Course4 
Course5 


### Computational Time - Activation Function
* A neuron will take an input, apply some activation function to it, and generate an output.
* Activation function plays an important role in computational time.
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/320px-Logistic-curve.svg_-300x200.png) <br/>
Logistic Sigmoid function - The slope, or the gradient of this function, at the extreme ends is close to zero. Therefore, the parameters are updated very slowly, resulting in very slow learning. Hence, switching from a sigmoid activation function to ReLU (Rectified Linear Unit) is one of the biggest breakthroughs we have seen in neural networks. ReLU updates the parameters much faster as the slope is 1 when x>0. This is the primary reason for faster computation of the models.
* But Rectified Linear Unit (ReLU) is 
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



