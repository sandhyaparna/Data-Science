Regression is a method of modelling a target value based on independent predictors.

### Gradient Descent Algorithm
A gradient measures how much the output of a function changes if you change the inputs a little bit. <br/>
Gradient descent is an optimization algorithm that finds the optimal weights (a,b) (Equation: Y=a+bX) that reduces prediction error i.e difference between actual and predicted values. Steps:
1. Initialize the weights(a & b) with random values and calculate Error (SSE)
2. Calculate the gradient i.e. change in SSE when the weights (a & b) are changed by a very small value from their original randomly initialized value. This helps us move the values of a & b in the direction in which SSE is minimized.
3. Adjust the weights with the gradients to reach the optimal values where SSE is minimized
4. Use the new weights for prediction and to calculate the new SSE
5. Repeat steps 2 and 3 till further adjustments to weights doesn’t significantly reduce the Error
Learning Rate: Determines how fast or slow we will move towards the optimal weights. In order for Gradient Descent to reach the local minimum, we have to set the learning rate to an appropriate value, which is neither too low nor too high. This is because if the steps it takes are too big, it maybe will not reach the local minimum and if you set the learning rate to a very small value, gradient descent will eventually reach the local minimum but it will maybe take too much time.
##### Batch Gradient Descent
It is also called vanilla gradient descent, calculates the error for each example within the training dataset, but only after all training examples have been evaluated, the model gets updated. This whole process is like a cycle and called a training epoch. <br/>
Parameters are updated only after evaluating all examples within training set are evaluated. Takes big, slow steps. <br/> 
It produces a stable error gradient and a stable convergence. <br/>
It is preferable for small datasets. <br/>
##### Stochastic Gradient Descent
Parameters are updated for each example within the training set, one by one. It takes small, quick steps. <br/>
The frequent updates (computationally expensive) allow us to have a pretty detailed rate of improvement. <br/>
It is preferable for large datasets. <br/>
##### Mini Batch Gradient Descent
It simply splits the training dataset into small batches and performs an update on the parameters for each of these batches. <br/> 
Common mini-batch sizes range between 50 and 256. <br/>


Sum of Squared Errors (SSE) = =  {Σ(Y – Ypred)^2} / 2




MSE - Mean Squared error


