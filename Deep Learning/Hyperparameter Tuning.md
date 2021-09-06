### Links
* https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8
* https://github.com/sandhyaparna/Data-Science/blob/e617d0d1a54293cd157d0ae65477c6a6329625e4/Deep%20Learning/Neural%20Networks.md
* GCP Model debubbing Techniques https://developers.google.com/machine-learning/testing-debugging/common/model-errors
* Interpreting Loss Curves https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic
* Hyperparameters AWS Doc https://docs.aws.amazon.com/machine-learning/latest/dg/training-parameters1.html

### Activation Functions
* https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
* Choosing an activation function
  * For multiple classification, use softmax on the output layer
  * RNN’s do well with Tanh
  * For everything else
  * Start with ReLU
  * If you need to do better, try Leaky ReLU
  * Last resort: PReLU , Maxout
  * Swish for really deep networks

### Summary of hyperparameter tuning
Most machine learning problems require a lot of hyperparameter tuning. Unfortunately, we can't provide concrete tuning rules for every model. Lowering the learning rate can help one model converge efficiently but make another model converge much too slowly. You must experiment to find the best set of hyperparameters for your dataset. That said, here are a few rules of thumb:
* Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
* If the training loss does not converge, train for more epochs.
* If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
* If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
* Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
* Setting the batch size to a very small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
* For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.
* Remember: the ideal combination of hyperparameters is data dependent, so you must always experiment and verify.

### Learning Rate
* Step size = (Gradient or slope) * (learning rate)
* If learning rate is small, loss keeps decreasing and finally converges, but if learning rate is high, loss oscillates/diverges. But lower learning rate takes more time. Large learning rates help to regularize the training but if the learning rate is too large, the training will diverge. 
* An oscillating loss curve strongly suggests that the learning rate is too high. https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=linear_regression_synthetic_tf2-colab&hl=en
* https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_a_real_dataset.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=linear_regression_real_tf2-colab&hl=en
* Training loss should steadily decrease, steeply at first, and then more slowly. Eventually, training loss should eventually stay steady (zero slope or nearly zero slope), which indicates that training has converged.
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
* Good to start with 128/256/512 for a sample of 1M observations. For small datasets use 32/
* larger batch sizes make larger gradient steps than smaller batch sizes
* Large batch sizes may result in different outputs during run and may fall into local mimima
* Use smaller batch size for consistent results
* Small batch sizes add regularization while large batch sizes add less, so utilize this while balancing the proper amount of regularization
* Small batch sizes tend to not get stuck in local minima
* Large batch sizes can converge on the wrong solution at random
* Large learning rates can overshoot the correct solution
* Small learning rates increase training time

### Overfitting
* Identify:
  * Underfit Model. A model that fails to sufficiently learn the problem and performs poorly on a training dataset and does not perform well on a holdout sample.
  * Overfit Model. A model that learns the training dataset too well, performing well on the training dataset but does not perform well on a hold out sample.
  * Good Fit Model. A model that suitably learns the training dataset and generalizes well to the old out dataset.
* Fix
  * Regularization
  * Dropout
  * Batch Normalization
  * Early stopping
  * Reduce overfitting by training the network on more examples.
  * Reduce overfitting by changing the complexity of the network. 
  * Recommendations for Multilayer Perceptrons and Convolutional Neural Networks
    * Classical: use early stopping and weight decay (L2 weight regularization).
    * Alternate: use early stopping and added noise with a weight constraint.
    * Modern: use early stopping and dropout, in addition to a weight constraint.
  * Recommendations for RNN
    * Classical: use early stopping with added weight noise and a weight constraint such as maximum norm.
    * Modern: use early stopping with a backpropagation-through-time-aware version of dropout and a weight constraint.

### Regularization
* Prevents overfitting
* L1 Lasso: Sparse Output, Computationally inefficient. L1 values between 1E-4 and 1E-8 have been found to produce good results. Larger values are likely to produce models that aren't very useful. You can't set both L1 and L2. You must choose one or the other.
* L2 Ridge: Dense Output, Computationally efficient. L2 values between 1E-2 and 1E-6 have been found to produce good results. Larger values are likely to produce models that aren't very useful.

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

### How to configure layers and neurons
* https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/
* The most reliable way to configure these hyperparameters for your specific predictive modeling problem is via systematic experimentation with a robust test harness.
* Number of layers:
  * Well if the data is linearly separable then you don't need any hidden layers at all. 
  * If data is less complex and is having fewer dimensions or features then neural networks with 1 to 2 hidden layers would work.
  * If data is having large dimensions or features then to get an optimum solution, 3 to 5 hidden layers can be used. 
* Number of Neurons
  * Depends on
    * number of input and output units 
    * number of training cases 
    * amount of noise in the targets 
    * complexity of the function or classification to be learned 
    * architecture 
    * type of hidden unit activation function 
    * training algorithm 
    * regularization 
  * The most appropriate number of hidden neurons is **sqrt(input layer nodes * output layer nodes)**. The number of hidden neurons should keep on decreasing in subsequent layers to get more and more close to pattern and feature extraction and to identify the target class.
  * The number of hidden neurons should be between the size of the input layer and the size of the output layer. mean of input and output layers
  * Overall Number of Neurons in hidden layer Nh=(Ns)/(α∗(Ni+No))
    * Ni = number of input neurons.
    * No = number of output neurons.
    * Ns = number of samples in training data set.
    * α = an arbitrary scaling factor usually 2-10. 
  * look at your weight matrix after training; look weights very close to zero and remove those. Get weights of layers in Tensorflow https://www.gcptutorials.com/post/how-to-get-weights-of-layers-in-tensorflow
  * The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
  * The number of hidden neurons should be less than twice the size of the input layer - you will never require more than twice the number of hidden units as you have inputs in an MLP with one hidden layer
   have inputs
* Layers:
  * Input Layer: Input variables, sometimes called the visible layer.
  * Hidden Layers: Layers of nodes between the input and output layers. There may be one or more of these layers.
  * Output Layer: A layer of nodes that produce the output variables
* Terms
  * Size: The number of nodes in the model.
  * Width: The number of nodes in a specific layer.
  * Depth: The number of layers in a neural network.
  * Capacity: The type or structure of functions that can be learned by a network configuration. Sometimes called “representational capacity“.
  * Architecture: The specific arrangement of the layers and nodes in the network.

### Epochs
* In general, data sets with only a few observations typically require more passes over the data to obtain higher model quality. Larger data sets often contain many similar data points, which eliminates the need for a large number of passes. The impact of choosing more data passes over your data is two-fold: model training takes longer, and it costs more

### Data Shuffling
* Shuffling mixes up the order of your data so that the SGD algorithm doesn't encounter one type of data for too many observations in succession. When model sees similar type of category data, updating of parameters might be difficult when it suddenly encounters new category. 
* You must shuffle your training data even if you chose the random split option when you split the input datasource into training and evaluation portions. The random split strategy chooses a random subset of the data for each datasource, but it doesn't change the order of the rows in the datasource. 







