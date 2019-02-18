### Links
https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526 </br>


### Overview
An embedding is a mapping of a discrete — categorical — variable to a vector of continuous numbers. In the context of neural networks, embeddings are low-dimensional, learned continuous vector representations of discrete variables. Neural network embeddings are useful because they can reduce the dimensionality of categorical variables and meaningfully represent categories in the transformed space. </br>
 </br>
Embeddings are feature columns that function like NN layers. Take a sparse vector encoding and pass it through an embedding column and then use that embedding column as the input, along with other features, to a DNN and to train the DNN. </br>
 </br>
Tensorflow can do math operations on sparse tensors without having to convert them into dense form i.e on-hot encoding. Foe eg: if different videos on Youtube are given ID from 1 to n. For each observation a sparse vector is created i.e if a person watched 1,5,7,10 IDs videos, that observation is represented as (1,5,7,10). To create embeddings, we take original input and represent it as a sparse tensor. Next, pass it through an embedding layer.  </br>
 </br>
Number of embedding dimenisons is higher implies more accuracy but greater the chance of overfitting, slow training.
A good start for Number of embedding dimensions = (Possible Values)^(1/4)
 </br>
Embeddings are used to:
* Manage sparse data
* Reduce dimensionality
* Increase model generalization
* Cluster observations - Finding nearest neighbors in the embedding space. These can be used to make recommendations based on user interests or cluster categories.
 </br>
Embeddings are useful for any categtorical column. Embeddings are used in Text & Image classification.
 </br>
Create reusable embeddings  








