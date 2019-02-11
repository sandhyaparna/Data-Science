TensorFlow is an open source, high performance, library for numerical computation.  <br/>
Tensordlow is a lazy evaluation model - write a DAG and then you run the DAG in the context of a session to get results
But in tf.eager the evaluation is immediate and it's not lazy but it is typically not used in production programs and used only for development. <br/>
TensorBoard is used to visulaize tensorfloe graphs <br/>
A variable is a tensor whose value is initialized and then the value gets changed as a program runs. <br/>
  
#### Tensorflow APIs
* Core Tensorflow Python API - Numeric processing code, add, subtract, divide, matrix multiply etc. creating variables, creating tensors, getting the shape, all the dimensions of a tensor, all that core basic numeric processing stuff.  <br/>
* Components useful when building custon NN models <br/>
tf.layers - a way to create a new layer of hidden neurons, with a ReLU activation function. <br/>
tf.losses - a way to compute cross entropy with Logits.  <br/>
tf.metrics - a way to compute the root mean square error and data as it comes in. <br/>
* tf.estimator - knows how to do distributed training, it knows how to evaluate, how to create a checkpoint, how to Save a model, how to set it up for serving. It comes with everything done in a sensible way, that fits most machine learning models in production. <br/>
 
#### Estimator API







