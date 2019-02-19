### Links
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

### Overview
A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. 
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

* RNNs can learn to use the past information, but in some cases the gap between the relevant information and the place that it’s needed is small, whereas in othercases it is very large.
* Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.

### LSTM Networks - kind of RNN
* Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.
* LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)
The repeating module in a standard RNN contains a single layer.
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
The repeating module in an LSTM contains four interacting layers.
*









