Regression is a method of modelling a target value based on independent predictors.

### Gradient Descent Algorithm
A gradient measures how much the output of a function changes if you change the inputs a little bit <br/>
Gradient descent is an optimization algorithm that finds the optimal weights (a,b) (Equation: Y=a+bX) that reduces prediction error i.e difference between actual and predicted values. Steps:
1. Initialize the weights(a & b) with random values and calculate Error (SSE)
2. Calculate the gradient i.e. change in SSE when the weights (a & b) are changed by a very small value from their original randomly initialized value. This helps us move the values of a & b in the direction in which SSE is minimized.
3. Adjust the weights with the gradients to reach the optimal values where SSE is minimized
4. Use the new weights for prediction and to calculate the new SSE
5. Repeat steps 2 and 3 till further adjustments to weights doesn’t significantly reduce the Error




Sum of Squared Errors (SSE) = =  {Σ(Y – Ypred)^2} / 2




MSE - Mean Squared error


