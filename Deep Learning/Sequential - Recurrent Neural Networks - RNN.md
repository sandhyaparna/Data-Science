### Links
http://colah.github.io/posts/2015-08-Understanding-LSTMs/ <br/>
https://www.youtube.com/watch?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&v=_h66BW-xNgk <br/> 


### Problems
* Sequential modeling - Predict next word in a sentence. To do sequence modeling, we need to:
  * Handle variable-length sequences (Not fixed lenegth as in Feed forrward NN)
  * Track long-term dependencies
  * Maintain information about order
  * Share parameters across the sequence


### Overview
A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. <br/> Backpropagation throgh time = Backpropagating errors at each individual time stamp and across time stamps <br/>
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png) <br/>
ht = f(ht-1,xt) = function of previous state & current input i.e passed through activation function <br/>
* RNNs can learn to use the past information, but in some cases the gap between the relevant information and the place that it’s needed is small, whereas in othercases it is very large.
* Unfortunately, as that gap grows, RNNs become unable to learn to connect the information - This is vanishing gradient problem

### LSTM Networks - kind of RNN
* Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.
* LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
* LSTM networks reply on a gated cell to track info throughout many time steps 
* Working:
  * LSTMs FORGET irrelevant parts of the previous state
  * Takes both prior info and curent input, proocess and selectively UPDATE cell state
  * Use OUTPUT gatet output certain parts of the cell state <br/> <br/>
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png) <br/>
The repeating module in a standard RNN contains a single layer like above <br/> <br/> <br/>
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
The repeating module in the above LSTM contains four interacting layers
* They maintain internal cell state ct
* They use structures called gates to add or remove info to/from cell state











