Embeddings are used to:
* Manage sparse data
* Reduce dimensionality
* Increase model generalization
* Cluster observations

Embeddings are used in Text & Image classification

Embeddings are useful for any categtorical column. 

Create reusable embeddings

Tensorflow can do math operations on sparse tensors without having to convert them into dense form i.e on-hot encoding. Foe eg: if different videos on Youtube are given ID from 1 to n. For each observation a sparse vector is created i.e if a person watched 1,5,7,10 IDs videos, that observation is represented as (1,5,7,10). To create embeddings, we take original input and represent it as a sparse tensor. Next, pass it through an embedding layer. 






