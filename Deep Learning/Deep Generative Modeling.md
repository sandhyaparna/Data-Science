### Links


### Overview - Unsupervized
* Generating new data - Systems that dont just look to extract patterns in data but goes a step beyond to use those patterns to learn the underlying distribution in the data and use it to generate brand new data
* Generative modeling is subset of unsupervised learning, 
* Goal is to take input training samples from one distribution and learn a model that represents that distribution. 
  * Density Estimation 
  * Sample Generation 
* Debiasing is an imp factor. Models should be capable of uncovering underlying latent variables in a dataset
* New or rare situations like pedistrians on roads in self driving problem should be learnt. So, to leverage generative models, detect outliers in the distribution and use outliers during training to improve even more

### Latent variable Models - Autoencoders and Variational Autoencoders (VAEs); Generative Adversarial Networks (GANs)
Latent variable - latent variables, as opposed to observable variables, are variables that are not directly observed but are rather inferred (through a mathematical model) from other variables that are observed (directly measured). Mathematical models that aim to explain observed variables in terms of latent variables are called latent variable models. 
* Hidden or Latent variables are like objects that they cannot directly observe but are generating shadows
* Observable variables are shadows
Can we learn true explanatory factors i.e latent variables from only observed data?

### Autoencoders
![](https://blog.keras.io/img/ae/autoencoder_schema.jpg) <br/>
* Encoder - Takes image as input (x), pass it through CNN, to create Low-dimensional latent variable, z
* Low-dimensional latent space, z is created to remove noise and find most meaningful features
* Back-propagation cannot be learnt as it is an unsupervised model. So, we try to recontruct the original data. 
* Decoder - Train the model to use these features to reconstruct the original data. Learns mapping back from latent,z, to a reconstructed observation, x1
* Now x1 is compared with x and hence this problem is turned into a supervised model. Comparing x1 & x gives a loss term that can be backpropagated and improve encoder models to extract rich latent features, so that decoder can produce more accurate images
* Autoencoding is a form of compression
* Smaller latent space will force a large training bottleneck and hence will result in noiser data, blurry images
* Bottleneck hidden layer forces network to learn a compressed latent representation
* Recontruction loss forces the latent representation to capture as much info about the data as possible
* Here latent space is a deterministic function i,e if same input is fed multiple time u get the same output everytime

### Variational Autoencoders
![](https://cdn-images-1.medium.com/max/1600/1*D4hg5tL1LOGI2QJdG9zQ3w.jpeg)
* Extension of autoencoders with a probabilistic spin
* Replacing intermediate latent space that was deterministic with a probabilistic or stochastic distribution
* Mean and std dev are computed deterministically, takes this as input to cpmpute stochastic z





