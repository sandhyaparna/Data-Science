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
Neural neworks are typically organized in layers. Layers are made up of a number of interconnected 'nodes' which contain an 'activation function'. Patterns are presented to the network via the 'input layer', which communicates to one or more 'hidden layers' where the actual processing is done via a system of weighted 'connections'. The hidden layers then link to an 'output layer'  <br/>

Output = Non-linearity/Activation function on (Weighted vector of input + Bias) <br/>
Loss function/Error = Based on Prediction & Actual values - for all the observations. Loss function for regression is MSE (exponential graph similar to SSE - when errors are large, more penalization implies more contribution to loss function and to minimize loss function we need to reduce those big errors) los loss or cross-entropy is used for classification.

Feature crosses help linear models work in nonlinear problems but unfortunately it cannot solve all the real world problems. Neural Networks are an alternative to feature crossing by combining features. Layers are used to combine features, another layer to combine our combinations and so on. <br/>

Neural Networks can be arbitarily complex. To increase hidden dimensions, I can add NEURONS. To increase function composition, I can add LAYERS - mapping from original feature space to some new convoluted feature space. If I have multiple labels for example, I can add OUTPUTS. <br/>


### Computational Time - Activation Function <br/>
* A neuron/node/Unit/perceptron will take an input(weighted sum of inputs), apply some activation function (non-linear) to it, and generate an output.
* The purpose of the activation function is to introduce non-linearity into the output of a neuron. This is important because most real world data is non linear and we want neurons to learn these non linear representations.
* Activation function plays an important role in computational time.
* Every activation function (or non-linearity) takes a single number and performs a certain fixed mathematical operation on it  Some of the activation functions:
  * Logistic Sigmoid - σ(x) = 1 / (1 + exp(−x)) - Probability distribution - it is especially used for models where we have to predict the probability as an output.
  * Hyperbolic Tangent - tanh(x) = 2σ(2x) − 1 = 2/(1+[(e)^-2x])-1  - The tanh function is mainly used classification between two classes.The advantage is that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.
  * Rectified Linear Unit - Most Popular - ReLU - f(x) = 0 for x<0, x for x>=0 
  * eLU - Exponential Linear Unit - aplha(e^x - 1) for x<0, x for x>=0  
  * Softplus = ln(1+e^x) - ReLU has been modified so that the training doesnt stop when x is 0
  * Leaky ReLU - 0.01(x) for x<0, x for x>=0
  * Parametric ReLU - Alpha(x) for x<0, x for x>=0
  * ReLU6 - 3 segments - min(max(0,x),6) - 0 for x<0, x for x>=0&x<6, 6 for x>6 
  * Softmax - Multi-class predictions. Add additional constrainst that total of outputs=1, which allows outputs to be interpreted as probability.
    * If we have both mutually exclusive labels and probabilities, we should use tf.nn.softmax_cross_entropy_with_logits_v2.
    * If the labels are mutually exclusive, the probabilities aren't, then we should use tf.nn.sparse_softmax_cross_entropy_with_logits. 
    * If our labels aren't mutually exclusive, we should use tf.nn.sigmoid_cross_entropy_with_logits.
![](https://cdn-images-1.medium.com/max/1600/1*RD0lIYqB5L2LrI2VTIZqGw.png) <br/>
![](https://cdn-images-1.medium.com/max/1600/1*ypsvQH7kvtI2BhzR2eT_Sw.png) <br/>
* Slope, or the gradient of Sigmoid function, at the extreme ends is close to zero. Therefore, the parameters are updated very slowly, resulting in very slow learning. Hence, switching from a sigmoid activation function to ReLU (Rectified Linear Unit) is one of the biggest breakthroughs we have seen in neural networks. ReLU updates the parameters much faster as the slope is 1 when x>0. This is the primary reason for faster computation of the models.
* Sigmoid takes a real-valued input and squashes it to range between 0 and 1 which is equal to the range for Probability
* tanh takes a real-valued input and squashes it to the range [-1, 1]
* ReLU takes a real-valued input and thresholds it at zero (replaces negative values with zero implies slope is 0 when x<0). That means any negative input given to the ReLU activation function turns the value into zero immediately in the graph, which in turns affects the resulting graph by not mapping the negative values appropriately.
* eLU - Exponential linear unit, range is [0,infinity)]  
* Softmax - The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities. If one of the inputs is small or negative, the softmax turns it into a small probability, and if an input is large, then it turns it into a large probability, but it will always remain between 0 and 1. The pros of using the Softplus function are, it's continuous and differentiable at zero, unlike the ReLu function. 
https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
* Leaky ReLU - Have piecewise linear function in +ve domain. In -ve domain they have non-zero slope specifically, 0.01. 
* PReLU - Alpha is a learned parameter from training along with other neural network parameters
* Randomized Leaky ReLUs - Instead of Alpha being trained, it is sampled from a uniform distribution randomly. This can have an effect similar to drop out since you technically have a different network for each value of Alpha. And therefore, it is making something similar to an ensemble. At test time, all the values of Alpha are averaged together to a deterministic value to use for predictions. 
* ReLU6 - Capped at 6 
* eLU - Approximately linear in the non-negative portion of the input space and is smooth, monotonic and most importantly, non-zero in the negative portion of the input.The main drawback of ELUs are that they are more compositionally expensive than ReLUs due to having to calculate the exponential. 

##### Why do we need non-linear activation functions?
Using linear activation is essentially pointless. The composition of two linear functions is itself a linear function, and unless we use some non-linear activations, we are not computing more interesting functions. That’s why most experts stick to using non-linear activation functions. Adding non-linear activation functions to neural networks stops layers from collapsing back into just a linear model. <br/>

##### Advantages & Disadvantages of diff Activation functions
* Sigmoid & Tanh were great choices because they were differentiable everywhere, monotonic, and smooth. However, problems such as saturation would occur due to either high or low input values to the functions, which would end up in the asymptotic Plateau to the function. Since the curve is almost flat at these points, the derivatives are very close to zero. Therefore, training of the weights would go very slow or even halt since the gradients were all very close to zero, which will result in very small step sizes down the hill during gradient descent.
* ReLU activation function is nonlinear, so you can get the complex modeling needed, and it doesn't have the saturation in the non-negative portion of the input space. However, due to the negative portion of the input space translating to a zero activation, ReLU layers could end up dying or no longer activating, which can also cause training to slow or stop.
* one of the ways to solve this problem is using another activation function called the exponential linear unit or ELU. It is approximately linear and the non-negative portion of the input space, and it's smooth, monotonic and most importantly, non-zero in the negative portion of the input space. The main drawback of ELUs are that they are more computationally expensive than ReLUs due to having the calculated exponential.

### Performace
Performance with rest to data
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Screenshot-from-2018-10-12-14-29-37-850x438.png) <br/>
As the amount of data increases, the performance of traditional learning algorithms, like SVM and logistic regression, does not improve by a whole lot. In fact, it tends to plateau after a certain point. In the case of neural networks, the performance of the model increases with an increase in the data you feed to the model. <br/>
There are basically three scales that drive a typical deep learning process: <br/>
1. Data <br/>
2. Computation Time <br/>
3. Algorithms <br/>

### Algorithms
#### Back-Propagation Algorithm - When there are more hidden layers
Minimize loss of entire training set. Loss is a function of network of weights. Find the weights that gives us the least loss<br/>

Abbreviated as BackProp. Initially all the edge weights are randomly assigned. For every input in the training dataset, the ANN is activated and its output is observed. This output is compared with the desired output that we already know, and the error is "propagated" back to the previous layer. This error is noted and the weights are "adjusted" accordingly. Weights are adjusted using Gradient Descent Optimization. This process is repeated until the output error is below a predetermined threshold (Loss Optimization). <br/>
* Stochastic gradient descent is an iterative learning algorithm that uses a training dataset to update a model.
* Weights are initialized randomly using the following code = {np.random.randn(No of Input Vars, Number of neurons in the first hidden layer, Number of neurons in the second hidden layer, etc) * 0.01} <br/>
randn generates random floats from a univariate Normal Distribution of mean 0 & Variance 1. <br/>
Random values are multiplied with 0.01 to initialize small weights. If we initialize large weights, the activation will be large, resulting in zero slope (in case of sigmoid and tanh activation function). Hence, learning will be slow. So we generally initialize small weights randomly. <br/>
* An observation is also called an instance, an input vector or a feature vector  <br/>


Common failure modes for Gradient Descent: <br/>
* Problem1: Gradients can Vanish <br/>
  * Insight: When using sigmoid or tanh activation functions throughout your hidden layers. As you begin to saturate you end up in the asymptotic regions of the function which begin to plateau, the slope is getting closer and closer to approximately zero. When you go backwards through the network during back prop, your gradient can become smaller and smaller because you're compounding all these small gradients until the gradient completely vanishes. When this happens your weights are no longer updating and therefore training grinds to a halt. <br/>
  * Solution: a) Use non saturating non-linear activation functions such as ReLUs, ELUs, etc; b) Parameter initialization - Initialize weights. This prevents the weights from shrinking to 0  c) LSTM, GRU <br/>
* Problem2: Gradients can explode - weights gets bigger & bigger <br/>
  * Insight: Happens for sequence models with long sequence lengths, learning rate can be a factor here because in our weight updates, remember we multiplied the gradient with the learning rate and then subtract that from the current weight. So, even if the grading isn't that big with a learning rate greater than one it can now become too big and cause problems for us and our network.  <br/>
  * Solution: Batch normalization <br/>
* Problem3: Layers can die <br/>
  * Insight: Use Tensorboard to monitor summaries during and after training of our Neural network model. Monitor fraction of zero weights in Tensorboard. <br/>
  * Solution: Lower your learning rates <br/>
 
### Gradient Descent Optimization Algorithms
* Improves on the stochastic gradient descent
* Best site: https://medium.com/datadriveninvestor/overview-of-different-optimizers-for-neural-networks-e0ed119440c3
#### Adagrad
Adagrad [9] is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing smaller updates
(i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features. For this reason, it is well-suited for dealing with sparse data.RMSprop and Adadelta have both been developed independently around the same time stemming from the need to resolve Adagrad's radically diminishing learning rates.
* Decay the learning rate for parameters in proportion to their update history (more updates means more decay). The smaller the gradient accumulated, leads to a bigger learning rate

#### Adam
Adaptive Moment Estimation (Adam) [14] is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients , similar to momentum. 
* Do everything that RMSProp does to solve the denominator decay problem of AdaGrad. In addition to that, use a cumulative history of gradients.

#### Rprop
Rprop combines the idea of only using the sign of the gradient with the idea of adapting the step size individually for each weight. So, instead of looking at the magnitude of the gradient, we’ll look at the step size that’s defined for that particular weight. And that step size adapts individually over time, so that we accelerate learning in the direction that we need. To adjust the step size for some weight, the following algorithm is used:
* First, we look at the signs of the last two gradients for the weight.
* If they have the same sign, that means, we’re going in the right direction, and should accelerate it by a small fraction, meaning we should increase the step size multiplicatively(e.g by a factor of 1.2). If they’re different, that means we did too large of a step and jumped over a local minima, thus we should decrease the step size multiplicatively(e.g. by a factor of 0.5).
* Then, we limit the step size between some two values. These values really depend on your application and dataset, good values that can be for default are 50 and a millionth, which is a good start.
* Now we can apply the weight update.
 
#### RMSProp 
Rprop doesn’t really work when we have very large datasets and need to perform mini-batch weights updates. The reason it doesn’t work is that it violates the central idea behind stochastic gradient descent, which is when we have small enough learning rate, it averages the gradients over successive mini-batches. Consider the weight, that gets the gradient 0.1 on nine mini-batches, and the gradient of -0.9 on tenths mini-batch. What we’d like is to those gradients to roughly cancel each other out, so that the stay approximately the same. But it’s not what happens with rprop. With rprop, we increment the weight 9 times and decrement only once, so the weight grows much larger.
The central idea of RMSprop is keep the moving average of the squared gradients for each weight. And then we divide the gradient by square root the mean square. Which is why it’s called RMSprop(root mean square)
* AdaGrad decays the learning rate very aggressively (as the denominator grows). As a result, after a while, the frequent parameters will start receiving very small updates because of the decayed learning rate. To avoid this why not decay the denominator and prevent its rapid growth.

### To-Do when model has High Bias
* Bigger Network - More hidden layers or more hidden units
* Train it longer - doesn't always help but it certainly never hurts
* More advanced optimization Algos

### To-Do when model has High Variance
* Get more data
* Use Regularization
* Data Augmentation
* Early stopping

### Hyperparameters
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/ <br/>
https://www.analyticsvidhya.com/blog/2018/11/neural-networks-hyperparameter-tuning-regularization-deeplearning/ <br/>
* Weight matrix and Bias vector are parameters
The major difference between parameters and hyperparameters is that parameters are learned by the model during the training time, while hyperparameters can be changed before training the model.
* Learning rate – ⍺ in Back-Propagation (Gradient Descent Algo)
* Number of iterations / epochs needed for training our model
* Batch size
* Number of hidden layers
* Units in each hidden layer
* Choice of activation function
* lambda - λ Types of regularization: L1, L2 (aka Weight Decay), Elastic Net, explicit: Dropout, and implicit: Data Augmentation, and Early Stopping.
* Dropout - Regularization technique to avoid overfitting
* Momentum
![](https://cdn-images-1.medium.com/max/1000/1*0kBzZebGAdmaD1MjlZ9wNA.png) <br/>
![](https://cdn-images-1.medium.com/max/800/0*mxbFsZ0QfeNiIQ2f.jpeg) <br/>
#### Hidden layers & Units
* Number of Neurons per layer determines the number of dimensions of the vector space of a Neural network
* Many hidden units within a layer with regularization techniques can increase accuracy. Smaller number of units may cause underfitting.
* Use large hidden layers, same size for all hidden layers 
* Using a first hidden layer which is larger than the input layer tends to work better.
* When using unsupervised pre-training, the layers should be made much bigger than when doing purely supervised optimization.
#### Regularization - Check Regression for more on this
###### L2 or Weight Decay
The sum of squares in the L2 regularization penalty discourages large weights in the weights matrix, preferring smaller ones. Why might we want to discourage large weight values? In short, by penalizing large weights, we can improve the ability to generalize, and thereby reduce overfitting. the larger a weight value is, the more influence it has on the output prediction. Dimensions with larger weight values can almost singlehandedly control the output prediction of the classifier (provided the weight value is large enough, of course) which will almost certainly lead to overfitting.
###### Dropout
Dropout is the probability of dropping a neuron temporarily from the network rather than keeping it turned on. Suppose we add a dropout of 0.5 to all these images. The model will randomly remove 50% of the units from each layer and we finally end up with a much simpler network. Dropout works by randomly dropping out unit activations in a network for a single gradient step. Dropout simiulates ensemble learning. <br/>
* Here we insert a layer that randomly disconnects nodes from the previous layer to the next layer, thereby ensuring that no single node is responsible for learning how to represent a given class. <br/>
* Generally, use a small dropout value of 20%-50% of neurons with 20% providing a good starting point. A probability too low has minimal effect and a value too high results in under-learning by the network. <br/>
* The more you dropout, the stronger the regularization. 0=No dropout, 0.2 is typical, 1= drop everything out, learns nothing
* Use a larger network. You are likely to get better performance when dropout is used on a larger network, giving the model more of an opportunity to learn independent representations. <br/>
Dropout acts as another form of Regularization. It forces data to flow down multiple paths so that there is a more even spread. It also simulates ensemble learning. Don't forget to scale the dropout activations by the inverse of the keep probability. We remove dropout during prediction/inference.  
###### Data augmentation
Data augmentation can be used to overcome small dataset limitations. It purposely perturbs training examples, changing their appearance slightly, before passing them into the network for training. The end result is that a network consistently sees “new” training data points generated from the original training data, partially alleviating the need for us to gather more training data (though in general, gathering more training data will rarely hurt your algorithm). data augmentation can help dramatically reduce overfitting, all the while ensuring that our model generalizes better to new input samples. Particularly helpful when the image dataset is small — such as the Flower quite small, having only 80 images per class for a total of 1,360 images. A general rule of thumb when applying deep learning to computer vision tasks is to have 1,000–5,000 examples per class, so we are certainly at a huge deficit here which can be overcome using Data Augmentation.
#### Learning rate - Try 0.01 (Cyclical Learning rates for training Neural Nets)
Python keras_lr_finder Package to implement cyclical lr https://github.com/surmenok/keras_lr_finder <br/>
implement cyclical lr python https://github.com/metachi/fastaiv2keras <br/>
Learning rate finder Python https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0 Check out coments section also <br/>
Learning rate controls the size of the step in weight space <br/>
Train a network starting from a low learning rate and increase the learning rate exponentially for every batch. Record the learning rate and training loss for every batch. Then, plot the loss and the learning rate.  <br/>
Popular default learning rates used in different papers using Adam optimizer https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ <br/>
![](https://cdn-images-1.medium.com/max/800/1*HVj_4LWemjvOWv-cQO9y9g.png) <br/>
Another way to look at these numbers is calculating the rate of change of the loss (a derivative of the loss function with respect to iteration number), then plot the change rate on the y-axis and the learning rate on the x-axis. Graph is smoothed out using simple moving average. <br/>
![](https://cdn-images-1.medium.com/max/800/1*87mKq_XomYyJE29l91K0dw.png) <br/>
* Low learning rate slows down the learning process but converges smoothly. 
* Larger learning rate speeds up the learning but may not converge.
* Usually a decaying Learning rate is preferred. (If we slowly reduce the learning rate over time, we might speed up the learning process. This process is called learning rate decay. Initially, when the learning rate is not very small, training will be faster. If we slowly reduce the learning rate, there is a higher chance of coming close to the global minima. Learning rate decay can be given as: α = [1 / (1 + decay_rate * epoch_number)] * α0

#### Momentum
Momentum helps to know the direction of the next step with the knowledge of the previous steps. It helps to prevent oscillations. A typical choice of momentum is between 0.5 to 0.9. <br/>
#### Number of epochs
* The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes through the training dataset.
* Number of epochs is the number of times the whole training data is shown to the network while training.
* One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters
* Increase the number of epochs until the validation accuracy starts decreasing even when training accuracy is increasing(overfitting).
* Learning Curve - Error or skill of the model on Y-axis vs Number of epochs along the x-axis
#### Batch size
* The batch size is a hyperparameter of gradient descent that controls the number of training samples to work through before the model’s internal parameters are updated. Number of samples that gradient is calculated on.
  * Batch Gradient Descent. Batch Size = Size of Training Set
  * Stochastic Gradient Descent. Batch Size = 1
  * Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
* Mini batch size is the number of sub samples given to the network after which parameter update happens.
* A good default for batch size might be 32. Also try 32, 64, 128, 256, and so on. 
* Google cloud Recomendation: 40-100 tends to be a good tange for batch size. Can go up to as high as 500 <br/>
* If too small, training will bounce around <br/>
* If too large, training will take a very long time <br/>
#### Example for diff between Epochs and Batches
* Assume you have a dataset with 200 samples (rows of data) and you choose a batch size of 5 and 1,000 epochs.
* This means that the dataset will be divided into 40 batches, each with five samples. The model weights will be updated after each batch of five samples.
* This also means that one epoch will involve 40 batches or 40 updates to the model.
* With 1,000 epochs, the model will be exposed to or pass through the whole dataset 1,000 times. That is a total of 40,000 batches during the entire training process.


#### Iterations
Iterations is the number of batches needed to complete one epoch. Iterations=Number of Training examples/Batch size
  
  
  
### Limitations - Use Bayesian Deep Learning
* Very data hungry
* Computationally intensive to train and deploy
* Easily fooled by adversarial examples
* blackbox models

 <br/>




