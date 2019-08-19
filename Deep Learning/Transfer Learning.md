### Articles
http://ruder.io/transfer-learning/index.html </br>

### Scenarios
* The feature spaces of the source and target domain are different, e.g. the documents are written in two different languages.
* The marginal probability distributions of source and target domain are different, e.g. the documents discuss different topics. 
* The label spaces between the two tasks are different, e.g. documents need to be assigned different labels in the target task.
* The conditional probability distributions of the source and target tasks are different, e.g. source and target documents are unbalanced with regard to their classes.

### Methods
When there is large amounts of new data all the prameters in layers can be re-trained whereas if less data is available only the last layer can be initialized to random weigts
* Pre-Training: Initial data set used to create a model
* Fine-Tuning: Using new data on the pre-trained model






