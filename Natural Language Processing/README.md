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
  * Recurrent NN (LSTM are more resilient to the vanishing and exploding gradient problem an deal with both left and right context)
  * CNN (For text it only operates in 2 dimensions, with filters only needing to be moved along the temporal dimension. CNNs are more parallelizable than RNNs, as the state at every timestep only depends on the local context ,via the convolution operation, rather than all past states as in the RNN. CNNs can be extended with wider receptive fields using dilated convolutions to capture a wider context. CNNs and LSTMs can also be combined and stacked and convolutions can be used to speed up an LSTM) 
  * Recursive NN - RNNs and CNNs both treat the language as a sequence. From a linguistic perspective, however, language is inherently hierarchical: Words are composed into higher-order phrases and clauses, which can themselves be recursively combined according to a set of production rules. The linguistically inspired idea of treating sentences as trees rather than as a sequence gives rise to recursive neural networks. Bottom up approach and not left-to-right or right-to-left. Not only RNNs and LSTMs can be extended to work with hierarchical structures. Word embeddings can be learned based not only on local but on grammatical context ; language models can generate words based on a syntactic stack ; and graph-convolutional neural networks can operate over a tree 
* Sequence to sequence models - Machine Translation. NLG tasks. Applications include generating a caption based on an image , text based on a table, and a description based on source code changes.   Generate an output sequence by predicting one word at a time. For continuency parsing and Named Entity recognition. Transformer 
* Attention Mechanism - Transformer, BERT, GPT-2 </br>
The main bottleneck of sequence-to-sequence learning is that it requires to compress the entire content of the source sequence into a fixed-size vector. Attention alleviates this by allowing the decoder to look back at the source sequence hidden states, which are then provided as a weighted average as additional input to the decoder. Attention is widely applicable and potentially useful for any task that requires making decisions based on certain parts of the input. It has been applied to consituency parsing , reading comprehension (Hermann et al., 2015), and one-shot learning, among many others. The input does not even need to be a sequence, but can consist of other representations as in the case of image captioning. A useful side-effect of attention is that it provides a rare---if only superficial---glimpse into the inner workings of the model by inspecting which parts of the input are relevant for a particular output based on the attention weights. Attention is also not restricted to just looking at the input sequence; self-attention can be used to look at the surrounding words in a sentence or document to obtain more contextually sensitive word representations. Multiple layers of self-attention are at the core of the Transformer architecture.
* Memory based networks
* Pretrained language models



### Word Embedding Approaches
* Word2Vec - Distributed representations
* GloVe
* ULMFit
* ELMo
* BERT
* RoBERTa - Robustly optimized Bert pretraining approach
* XLNet
* ALBERT - A Lite BERT

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










