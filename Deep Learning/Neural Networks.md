

### Computational Time - Activation Function
* A neuron will take an input, apply some activation function to it, and generate an output.
* Activation function plays an important role in computationla time.
* Rectified Linear Unit (ReLU) is one of the most commonly used activation function.
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/320px-Logistic-curve.svg_-300x200.png) <br/>
Sigmoid function - The slope, or the gradient of this function, at the extreme ends is close to zero. Therefore, the parameters are updated very slowly, resulting in very slow learning. Hence, switching from a sigmoid activation function to ReLU (Rectified Linear Unit) is one of the biggest breakthroughs we have seen in neural networks. ReLU updates the parameters much faster as the slope is 1 when x>0. This is the primary reason for faster computation of the models.

### Performace
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Screenshot-from-2018-10-12-14-29-37-850x438.png) <br/>
As the amount of data increases, the performance of traditional learning algorithms, like SVM and logistic regression, does not improve by a whole lot. In fact, it tends to plateau after a certain point. In the case of neural networks, the performance of the model increases with an increase in the data you feed to the model. <br/>
There are basically three scales that drive a typical deep learning process: <br/>
1. Data <br/>
2. Computation Time <br/>
3. Algorithms <br/>











