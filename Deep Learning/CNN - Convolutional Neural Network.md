### Links
https://www.youtube.com/watch?v=H-HVZJ7kGI0&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=4&t=201s <br/>

### Applications
* Semantic Segmentation
* Object Detection
* Image Captioning
* Self driving cars
* Identifying facial phenotypes of genetic disorders

### Overview
Image classification - Spatial recognition

* Images have a lot of variation: 
  * Viewpoint variation
  * Illumination conditions
  * Scale variations
  * Deformation
  * Background clutter
  * Occlusion
  * Intr-class variation
* Models needs to be invariation to these above variations while still being sensitive to the differences that define the individual classes. Hence, features should be directly learnt from the data instead of hand engineering
* Visual features eg in a face
  * Low level features - Edges, dark spots
  * Eyes, ears, nose
  * Facial structure
* In NN, when Image is inputed as a vector of pixel values, there is no spatial information but visual data are rich in spatial structure and hence NN fails 

### Learning Visual features
In NN, when Image is inputed as a vector of pixel values, there is no spatial information but visual data are rich in spatial structure 
* Input: 2D image as an array of pixel values, patches of input should be connected to hidden layer i.e each neuron in the hidden layer sees only a particular region that is inputed
* Pixels that are spatially close to each other are more related to each other - Sliding window is created, to create sliding patches and take into account spatial info <br/>

Principle:  <br/>
-- Filter a size of 4 by 4 <br/>
-- Apply this same filter to 4 by 4 patches in input <br/>
-- Shift by 2 pixels for next patch <br/>
<br/>
This patchy operation is CONVOLUTION <br/>
* Apply a set of weights - a filter - to extract local features i.e each patch is multiplied with a filter (elementwise multiplication and the add the values in the obtained matrix to get a single number - Fill feature map using the obtained numbers
* Use multiple filters to extract different features i.e different weight filters are used to create distinct feature maps
* Spatially share parameters of each filter
By changing weights of filters we can detect and extract different filters

### CNNs for classification
Goal is to learn features directly from image data, i.e learning weights of those filters and using these learnt feature maps for image classification
Steps: 
* Convolution - Apply filters with learned weights to generate feature maps
* Non-linearity - Often ReLU
* Pooling - Downsampling operation on each feature map <br/>
Operation: 
* Each neuron take inputs from patch, compute weighted sum, apply bias
* Within a single convolutional layer, we can have many different filters that are learnt and hence output layer of a convolutional layer will have volume. d is number of filters, h & w are spatial dimensions
* Reduce dimensionality using pooling
* CONV and POOL layers output high-level features of input
* Fully connected layer uses these features for classifying input image
* Express output as probability of image belonging to a particular class

![](https://cdn-images-1.medium.com/max/1255/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg)
Feature learning part is the essence of CNN












