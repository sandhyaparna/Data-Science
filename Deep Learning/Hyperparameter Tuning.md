### Links
* https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8
* https://github.com/sandhyaparna/Data-Science/blob/e617d0d1a54293cd157d0ae65477c6a6329625e4/Deep%20Learning/Neural%20Networks.md



### Learning Rate
* Step size = (Gradient or slope) * (learning rate)
* If learning rate is small, loss keeps decreasing and finally converges, but if learning rate is high, loss oscillates/diverges. But lower learning rate takes more time. Large learning rates help to regularize the training but if the learning rate is too large, the training will diverge. 
![](https://github.com/sandhyaparna/Data-Science/blob/master/Deep%20Learning/Images/learning%20rate.png)

Run Code: code with example in https://github.com/sandhyaparna/Data-Science/blob/master/Deep%20Learning/Images/Implementing_LR_range_test_Learning_rate_finder.ipynb
* Run the function here: https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py
* Then run code in README of https://github.com/surmenok/keras_lr_finder

### Optimzation Algos
* Gradient Descent with Momentum: With each iteration of gradient descent, we move towards the local optima with up and down oscillations. If we use larger learning rate then the vertical oscillation will have higher magnitude. So, this vertical oscillation slows down our gradient descent and prevents us from using a much larger learning rate. Additionally, too small a learning rate makes the gradient descent slower.**We want a slower learning in the vertical direction and a faster learning in the horizontal direction which will help us to reach the global minima much faster**


### Weight Decay
* Weight decay is one form of regularization and it plays an important role in training so its value needs to be set properly [7]. Weight decay is defined as multiplying each weight in the gradient descent at each epoch by a factor λ [0<λ<1].
* Smaller datasets and architectures seem to require larger values for weight decay while larger datasets and deeper architectures seem to require smaller values. Our hypothesis is that complex data provides its own regularization and other regularization should be reduced.
* The optimal weight decay is different if you search with a constant learning rate versus using a learning rate range. This aligns with our intuition because the larger learning rates provide regularization so a smaller weight decay value is optimal.

### Batch size
* larger batch sizes make larger gradient steps than smaller batch sizes
* Large batch sizes may result in different outputs during run and may fall into local mimima
* Use smaller batch size for consistent results
* Small batch sizes add regularization while large batch sizes add less, so utilize this while balancing the proper amount of regularization
* Small batch sizes tend to not get stuck in local minima
* Large batch sizes can converge on the wrong solution at random
* Large learning rates can overshoot the correct solution
* Small learning rates increase training time

### Regularization
* Prevents overfitting
* L1: Sparse Output, Computationally inefficient
* L2: Dense Output, Computationally efficient

### Batch Normalization
* Used for overfitting - https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
* Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.  This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training. standardization refers to rescaling data to have a mean of zero and a standard deviation of one, e.g. a standard Gaussian. This process is also called “whitening” when applied to images in computer vision.
* Deep neural networks are challenging to train, not least because the input from prior layers can change after weight updates.
* Batch normalization is a technique to standardize the inputs to a network, applied to ether the activations of a prior layer or inputs directly.
* Batch normalization accelerates training, in some cases by halving the epochs or better, and provides some regularization, reducing generalization error. 
* Normalizing the inputs to the layer has an effect on the training of the model, dramatically reducing the number of epochs required. It can also have a regularizing effect, reducing generalization error much like the use of activation regularization.
* It may be more appropriate **after** the activation function if for s-shaped functions like the hyperbolic tangent and logistic function.
* It may be appropriate **before** the activation function for activations that may result in non-Gaussian distributions like the rectified linear activation function, the modern default for most network types.
* Using batch normalization makes the network more stable during training. **This may require the use of much larger than normal learning rates, that in turn may further speed up the learning process.** The faster training also means that the decay rate used for the learning rate may be increased.
* Deep neural networks can be quite sensitive to the technique used to initialize the weights prior to training. But the stability to training brought by batch normalization can make training deep networks less sensitive to the choice of weight initialization method.
* It may not be appropriate for variables that have a data distribution that is highly non-Gaussian, in which case it might be better to perform data scaling as a pre-processing step.
* Batch normalization offers some regularization effect, reducing generalization error, perhaps **no longer requiring the use of dropout** for regularization. The reason is that the statistics used to normalize the activations of the prior layer may become noisy given the random dropping out of nodes during the dropout procedure.

### Vanishing Gradient Problem 
* Check derivatives computed during training
* Use LSTM instead of RNN, ResNet instead of CNN
* Use ReLu
* Use Multi-level hierarchy: Break up levels into their own sub-networks trained individually
* Ensemble of shorter networks

### Exploding gradient
* Identify Exploding gradients:
  * The model is unable to get traction on your training data (e.g. poor loss).
  * The model is unstable, resulting in large changes in loss from update to update.
  * The model loss goes to NaN during training.
  * The model weights quickly become very large during training
  * The model weights go to NaN values during training.
  * The error gradient values are consistently above 1.0 for each node and layer during training.
* Fix
  * Re-Design the Network Model: redesigning the network to have fewer layers. **use a smaller batch size**
  * **Gradient clipping**: Exploding gradients can still occur in very deep Multilayer Perceptron networks with a large batch size and LSTMs with very long input sequence lengths. If exploding gradients are still occurring, you can check for and limit the size of gradients during the training of your network. Specifically, the values of the error gradient are checked against a threshold value and clipped or set to that threshold value if the error gradient exceeds the threshold.In the Keras deep learning library, you can **use gradient clipping by setting the clipnorm or clipvalue arguments on your optimizer** before training. **Good default values are clipnorm=1.0 and clipvalue=0.5**
  * weight regularization: and often an L1 (absolute weights) or an L2 (squared weights) penalty can be used

### Wide vs Deep Models:
* Wide (more neurons in a layer) overfits. Model memorizes training points
* A sufficiently wide neural network with just a single hidden layer can approximate any (reasonable) function given enough training data. There are, however, a few difficulties with using an extremely wide, shallow network. **The main issue is that these very wide, shallow networks are very good at memorization, but not so good at generalization**. So, if you train the network with every possible input value, a super wide network could eventually memorize the corresponding output value that you want. But that's not useful because for any practical application you won't have every possible input value to train with.
* **The advantage of multiple layers is that they can learn features at various levels of abstraction.** For example, if you train a deep convolutional neural network to classify images, you will find that the first layer will train itself to recognize very basic things like edges, the next layer will train itself to recognize collections of edges such as shapes, the next layer will train itself to recognize collections of shapes like eyes or noses, and the next layer will learn even higher-order features like faces. **Multiple layers are much better at generalizing because they learn all the intermediate features between the raw data and the high-level classification.**
* If you build a very wide, very deep network, you run the chance of each layer just memorizing what you want the output to be, and you end up with a neural network that fails to generalize to new data.






