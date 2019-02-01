### Links
Course1 https://www.analyticsvidhya.com/blog/2018/10/introduction-neural-networks-deep-learning/ <br/>
Course2 https://www.analyticsvidhya.com/blog/2018/11/neural-networks-hyperparameter-tuning-regularization-deeplearning/ <br/>
Course3  <br/>
Course4 https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/ <br/>
Course5 https://www.analyticsvidhya.com/blog/2019/01/sequence-models-deeplearning/ <br/>
Intro to NN https://www.kdnuggets.com/2016/11/quick-introduction-neural-networks.html

### Intro
Neural Networks takes input features, automatically identifies hidden features from input and finally generates the output. It is inspired by the way biological neural networks in the human brain process information. <br/>
Def: A computing system made up of a number of simple, highly interconnected processing elements, which process information by their dynamic state response to external inputs <br/>
Neural neworks are typically organized in layers. Layers are made up of a number of interconnected 'nodes' which contain an 'activation function'. Patterns are presented to the network via the 'input layer', which communicates to one or more 'hidden layers' where the actual processing is done via a system of weighted 'connections'. The hidden layers then link to an 'output layer' 
### Computational Time - Activation Function <br/>
* A neuron/node/Unit will take an input, apply some activation function (non-linear) to it, and generate an output.
* The purpose of the activation function is to introduce non-linearity into the output of a neuron. This is important because most real world data is non linear and we want neurons to learn these non linear representations.
* Activation function plays an important role in computational time.
* Every activation function (or non-linearity) takes a single number and performs a certain fixed mathematical operation on it  Some of the activation functions:
  * Logistic Sigmoid - σ(x) = 1 / (1 + exp(−x))
  * Hyperbolic Tangent - tanh(x) = 2σ(2x) − 1
  * Rectified Linear Unit - Most Popular - ReLU - f(x) = max(0, x)
  * Softplus
  * Softmax - Multi-class predictions
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
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/ <br/>
https://www.analyticsvidhya.com/blog/2018/11/neural-networks-hyperparameter-tuning-regularization-deeplearning/ <br/>
* Weight matrix and Bias vector are parameters
The major difference between parameters and hyperparameters is that parameters are learned by the model during the training time, while hyperparameters can be changed before training the model.
* Learning rate – ⍺ in Back-Propagation (Gradient Descent Algo)
* Number of iterations / epochs needed for training our model
* Number of hidden layers
* Units in each hidden layer
* Choice of activation function
* lambda - λ Types of regularization: L1, L2 (aka Weight Decay), Elastic Net, explicit: Dropout, and implicit: Data Augmentation, and Early Stopping.
* Dropout - Regularization technique to avoid overfitting
* Momentum
![](https://cdn-images-1.medium.com/max/1000/1*0kBzZebGAdmaD1MjlZ9wNA.png) <br/>
![](https://cdn-images-1.medium.com/max/800/0*mxbFsZ0QfeNiIQ2f.jpeg) <br/>
#### Hidden layers & Units
* Many hidden units within a layer with regularization techniques can increase accuracy. Smaller number of units may cause underfitting.
* Use large hidden layers, same size for all hidden layers 
* Using a first hidden layer which is larger than the input layer tends to work better.
* When using unsupervised pre-training, the layers should be made much bigger than when doing purely supervised optimization.
#### Regularization
###### L2 or Weight Decay
The sum of squares in the L2 regularization penalty discourages large weights in the weights matrix, preferring smaller ones. Why might we want to discourage large weight values? In short, by penalizing large weights, we can improve the ability to generalize, and thereby reduce overfitting. the larger a weight value is, the more influence it has on the output prediction. Dimensions with larger weight values can almost singlehandedly control the output prediction of the classifier (provided the weight value is large enough, of course) which will almost certainly lead to overfitting.
###### Dropout
Here we insert a layer that randomly disconnects nodes from the previous layer to the next layer, thereby ensuring that no single node is responsible for learning how to represent a given class. <br/>
Generally, use a small dropout value of 20%-50% of neurons with 20% providing a good starting point. A probability too low has minimal effect and a value too high results in under-learning by the network. <br/>
* Use a larger network. You are likely to get better performance when dropout is used on a larger network, giving the model more of an opportunity to learn independent representations.
###### Data augmentation
Data augmentation can be used to overcome small dataset limitations. It purposely perturbs training examples, changing their appearance slightly, before passing them into the network for training. The end result is that a network consistently sees “new” training data points generated from the original training data, partially alleviating the need for us to gather more training data (though in general, gathering more training data will rarely hurt your algorithm). data augmentation can help dramatically reduce overfitting, all the while ensuring that our model generalizes better to new input samples. Particularly helpful when the image dataset is small — such as the Flower quite small, having only 80 images per class for a total of 1,360 images. A general rule of thumb when applying deep learning to computer vision tasks is to have 1,000–5,000 examples per class, so we are certainly at a huge deficit here which can be overcome using Data Augmentation.
#### Learning rate - Try 0.01 (Cyclical Learning rates for training Neural Nets)
Python keras_lr_finder Package to implement cyclical lr https://github.com/surmenok/keras_lr_finder <br/>
implement cyclical lr python https://github.com/metachi/fastaiv2keras <br/>
Learning rate finder Python https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0 Check out coments section also <br/>
Train a network starting from a low learning rate and increase the learning rate exponentially for every batch. Record the learning rate and training loss for every batch. Then, plot the loss and the learning rate.  <br/>
![](https://cdn-images-1.medium.com/max/800/1*HVj_4LWemjvOWv-cQO9y9g.png) <br/>
Another way to look at these numbers is calculating the rate of change of the loss (a derivative of the loss function with respect to iteration number), then plot the change rate on the y-axis and the learning rate on the x-axis. Graph is smoothed out using simple moving average. <br/>
![](https://cdn-images-1.medium.com/max/800/1*87mKq_XomYyJE29l91K0dw.png) <br/>
* Low learning rate slows down the learning process but converges smoothly. 
* Larger learning rate speeds up the learning but may not converge.
* Usually a decaying Learning rate is preferred.
#### Momentum
Momentum helps to know the direction of the next step with the knowledge of the previous steps. It helps to prevent oscillations. A typical choice of momentum is between 0.5 to 0.9. <br/>
#### Number of epochs
* Number of epochs is the number of times the whole training data is shown to the network while training.
* Increase the number of epochs until the validation accuracy starts decreasing even when training accuracy is increasing(overfitting).
#### Batch size
* Mini batch size is the number of sub samples given to the network after which parameter update happens.
* A good default for batch size might be 32. Also try 32, 64, 128, 256, and so on. 
#### Iterations
Iterations is the number of batches needed to complete one epoch. Iterations=Number of Training examples/Batch size


 <br/>




