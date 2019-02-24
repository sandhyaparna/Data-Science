### Links
https://www.youtube.com/watch?v=H-HVZJ7kGI0&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=4&t=201s <br/>



Image classification

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
* Pixels that are spatially close to each other are more related to each other - Sliding window is created, to create sliding patches and take into account spatial info
 <br/>
Principle: 
* Apply a set of weights - a filter - to extract local features
* Use multiple filters to extract different features
* Spatially share parameters of each filter






Spatial recognition







