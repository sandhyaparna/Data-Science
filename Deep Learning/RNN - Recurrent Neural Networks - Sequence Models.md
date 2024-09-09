### Links
Sequence Models by Andrew Ng https://www.coursera.org/learn/nlp-sequence-models/lecture/0h7gT/why-sequence-models <br/>
Analytics Vidhya Notes https://www.analyticsvidhya.com/blog/2019/01/sequence-models-deeplearning/ <br/>
MIT https://www.youtube.com/watch?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&v=_h66BW-xNgk <br/>

LSTM http://blog.echen.me/ <br/> 
LSTM http://colah.github.io/posts/2015-08-Understanding-LSTMs/ <br/>
GRU https://medium.com/@anishnama20/understanding-gated-recurrent-unit-gru-in-deep-learning-2e54923f3e2 <br/>

### Applications
* Sequential modeling - Predict next word in a sentence. To do sequence modeling, we need to:
  * Handle variable-length sequences (Not fixed lenegth as in Feed forrward NN)
  * Track long-term dependencies
  * Maintain information about order
  * Share parameters across the sequence
* Speech Recognition
* Music generation - Train on old music to generate brand new music
* Sentiment classification
* Machine translation - Attention mechanisms
* NER
</br>
Most of the tasks in NLP such as text classification, language modeling, machine translation, etc. are sequence modeling tasks. The traditional machine learning models and neural networks cannot capture the sequential information present in the text. Therefore, people started using recurrent neural networks (RNN and LSTM) because these architectures can model sequential information present in the text.



### Overview
A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. <br/> Backpropagation through time = Backpropagating errors at each individual time stamp and across time stamps <br/>
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png) <br/>
ht = f(ht-1,xt) = function of previous state & current input i.e passed through activation function <br/>
* RNNs can learn to use the past information, but in some cases the gap between the relevant information and the place that it’s needed is small, whereas in othercases it is very large.
* Unfortunately, as that gap grows, RNNs become unable to learn to connect the information - This is vanishing gradient problem

### LSTM Networks - kind of RNN
* Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.
* LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!
* LSTM networks reply on a gated cell to track info throughout many time steps <br/> <br/>
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png) <br/>
The repeating module in a standard RNN contains a single layer like above <br/> <br/> <br/>
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
The repeating module in the above LSTM contains four interacting layers
* They maintain internal cell state ct
* They use structures called gates to control flow of info (add or remove info to/from cell state) - Model long-term dependencies https://medium.com/@purnasaigudikandula/recurrent-neural-networks-and-lstm-explained-7f51c7f6bbb9
  * LSTMs FORGET irrelevant parts of the previous state
  * Takes both prior info and curent input, proocess and selectively UPDATE cell state
  * Use OUTPUT gatet output certain parts of the cell state 
* Back propagation from ct to ct-1 doesn't require matrix multiplication: uninterrupted gradient flow
* There are three steps in an LSTM network:
  * Step 1: The network decides what to forget and what to remember.
  * Step 2: It selectively updates cell state values.
  * Step 3: The network decides what part of the current state makes it to the output.

### GRU - Gated Recurrent Unit
* GRU stands for Gated Recurrent Unit, which is a type of recurrent neural network (RNN) architecture that is similar to LSTM (Long Short-Term Memory).
* Like LSTM, GRU is designed to model sequential data by allowing information to be selectively remembered or forgotten over time. However, GRU has a simpler architecture than LSTM, with fewer parameters, which can make it easier to train and more computationally efficient.
* The main difference between GRU and LSTM is the way they handle the memory cell state. In LSTM, the memory cell state is maintained separately from the hidden state and is updated using three gates: the input gate, output gate, and forget gate. In GRU, the memory cell state is replaced with a “candidate activation vector,” which is updated using two gates: the reset gate and update gate.
* The reset gate determines how much of the previous hidden state to forget, while the update gate determines how much of the candidate activation vector to incorporate into the new hidden state.
* Overall, GRU is a popular alternative to LSTM for modeling sequential data, especially in cases where computational resources are limited or where a simpler architecture is desired.








