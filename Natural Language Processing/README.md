### Blogs/Articles
* http://ruder.io/

### Applications
* Spam Detection
* POS Tagging
* Sentiment Analyzer
* Machine Translation (Text or speech of 1 language into another) - Seq2Seq learning - Text Generation
* Latent Semantic Analysis (LSA)
* Article Spinning
* Machine Conversations (Text-to-Speech & Speech-to-Text conversion)
* Paraphrasing and summarization
* Skim Reading
* Topic Discovery (LDA) & Modeling 
* Chatbots

### Natural Language Understanding
* Phonology – This science helps to deal with patterns present in the sound and speeches related to the sound as a physical entity. 
* Pragmatics – This science studies the different uses of language.
* Morphology – This science deals with the structure of the words and the systematic relations between them.
* Syntax – This science deal with the structure of the sentences.
* Semantics – This science deals with the literal meaning of the words, phrases as well as sentences.

### Advances in NLP - https://ruder.io/a-review-of-the-recent-history-of-nlp/
* Neural Language Models - Language modelling is the task of predicting the next word in a text given the previous words. Egs - Intelligent keyboards, email response suggestion, spelling autocorrection, etc. Approaches are n-grams. Started with feed forward NN and currently uses RNNs, LSTM. 
* Multi-task Learning - Method for sharing parameters between models that are trained on multiple tasks. In neural networks, this can be done easily by tying the weights of different layers.
* Word embeddings - word2vec. Training on a very large corpus enables them to approximate certain relations between words such as gender, verb tense, and country-capital relations. </br>
  * Later Studies showed that there is nothing inherently special about word2vec: Word embeddings can also be learned via matrix factorization and with proper tuning, classic matrix factorization approaches like SVD and LSA achieve similar results.
* Neural Networks for NLP - 
  * Recurrent NN (LSTM are more resilient to the vanishing and exploding gradient problem. A bidirectional LSTM is typically used to deal with both left and right context)
  * CNN (For text it only operates in 2 dimensions, with filters only needing to be moved along the temporal dimension. CNNs are more parallelizable than RNNs, as the state at every timestep only depends on the local context ,via the convolution operation, rather than all past states as in the RNN. CNNs can be extended with wider receptive fields using dilated convolutions to capture a wider context. CNNs and LSTMs can also be combined and stacked and convolutions can be used to speed up an LSTM) 
  * Recursive NN - RNNs and CNNs both treat the language as a sequence. From a linguistic perspective, however, language is inherently hierarchical: Words are composed into higher-order phrases and clauses, which can themselves be recursively combined according to a set of production rules. The linguistically inspired idea of treating sentences as trees rather than as a sequence gives rise to recursive neural networks. Bottom up approach and not left-to-right or right-to-left. Not only RNNs and LSTMs can be extended to work with hierarchical structures. Word embeddings can be learned based not only on local but on grammatical context ; language models can generate words based on a syntactic stack ; and graph-convolutional neural networks can operate over a tree 
* Sequence to sequence models -
  * https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
  * A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images…etc) and outputs another sequence of items. In neural machine translation, a sequence is a series of words, processed one after another. The output is, likewise, a series of words. 
  * Model is composed of an encoder and decoder. The encoder processes each item in the input sequence, it compiles the information it captures into a vector (called the context i.e an array of numbers). After processing the entire input sequence, the encoder sends the context over to the decoder, which begins producing the output sequence item by item. Size of the context vector is the number of hidden units in the encoder. Encoder takes 2 inputs at each step: one word from the input sentence i.e represented as vector (Word is trandformed into a vector using word embedding algos) and a hidden state. We get an updated hidden state in the output from first step. In the 2nd step of encoding 2nd word and hidden step output from first step are sent as input. And only the last hidden state from Encoding stage is sent to every step of Decoding stage. Every step of decoding stage takes 2 inputs: Last hidden state from encoding stage and hidden state obtained from the previous step of decoding 
  * Applications include Machine Transalation, generating a caption based on an image , text based on a table, text summarization, and a description based on source code changes.   Generate an output sequence by predicting one word at a time. For continuency parsing and Named Entity recognition. Transformer 
* Attention Mechanism - Transformer networks - BERT, GPT-2 </br>
  * Instead of passing the last hidden state of the encoding stage, the encoder passes all the hidden states to the decoder
  * Second, an attention decoder does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the decoder does the following:
    * Look at the set of encoder hidden states it received – each encoder hidden states is most associated with a certain word in the input sentence
    * Give each hidden states a score (let’s ignore how the scoring is done for now)
    * Multiply each hidden states by its softmaxed score, thus amplifying hidden states with high scores, and drowning out hidden states with low scores
  * The main bottleneck of sequence-to-sequence learning is that it requires to compress the entire content of the source sequence into a fixed-size vector. Attention alleviates this by allowing the decoder to look back at the source sequence hidden states, which are then provided as a weighted average as additional input to the decoder. Attention is widely applicable and potentially useful for any task that requires making decisions based on certain parts of the input. It has been applied to consituency parsing , reading comprehension (Hermann et al., 2015), and one-shot learning, among many others. The input does not even need to be a sequence, but can consist of other representations as in the case of image captioning. A useful side-effect of attention is that it provides a rare---if only superficial---glimpse into the inner workings of the model by inspecting which parts of the input are relevant for a particular output based on the attention weights. Attention is also not restricted to just looking at the input sequence; self-attention can be used to look at the surrounding words in a sentence or document to obtain more contextually sensitive word representations. Multiple layers of self-attention are at the core of the Transformer architecture.
* Memory based networks
* Pretrained language models
* Transfer Learning: The use of models trained on a particular domain of learning tasks and repurposing the learned weights to solve another similar learning task is called transfer learning. Fine-tuning is the process of updating weights of a pre-trained model

### Word Embedding Approaches
* Word2Vec - Distributed representations
* GloVe
* FastText

### Pre-trained language models:
* ULMFit (FastAI) - Transfer Learning Technique
* ELMo
* BERT (PyTorch Transformers, HuggingFace Transformers) 
* Transformer-XL
* Stanford NLP
* OpenAI's GPT-2
* XLNet
* PyTorch - Transformers
* Baidu's Enhanced Representation through knowledge Intergration
* RoBERTa - Robustly optimized Bert pretraining approach
* FacebookAI's XLM/mBERT
* ALBERT - A Lite BERT
* REALM - Retrieval-Augmented Language Model Pre-Training 



### Annotation tools
* Prodigy
* Doccano



### Algos
* ConZNet - Deep reinforcement algorithm
* PyTorch Transformers https://github.com/huggingface/pytorch-transformers

### Libraries
* Spark NLP
* TensorFlow
* Keras
* PyTorch
* Spark NLP
* spaCy
* AllenNLP
* FastText - FB's NLP lib
* MOE model- misspelling oblivious embeddings
* AllenAI scispacy

##### Less Accurate 
* spaCy
* Stanford CoreNLP
* nltk
* OpenNLP

#### LinkedIn
https://www.linkedin.com/posts/stevenouri_natural-language-processing-nlp-with-python-ugcPost-6594922762333229056-tHXk










