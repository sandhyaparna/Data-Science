### Links
NLP 100hrs http://www.supervisedlearning.com/nlp100hours <br/>
KDNuggets NLP in Nutshell https://www.kdnuggets.com/2019/01/solve-90-nlp-problems-step-by-step-guide.html <br/>
Readmission by Andrew https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709 <br/>
https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/ <br/>
https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/ <br/>
https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html <br/>
https://www.nltk.org/book/ch06.html <br/>
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html <br/>
https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/ <br/>
https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/ <br/>
https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa <br/>
https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41 <br/>
GloVe - Pretrained https://nlp.stanford.edu/projects/glove/ <br/>
NLP Repo https://lnkd.in/fyyvZYt <br/>


### Pre-processing
Tokenization – Process of converting a text into tokens
Tokens – Words or entities present in the text
Text object – A sentence or a phrase or a word or an article
##### Steps
* Noise Removal - Stopwords, Punctuations, URLs or links, social media entities(mentions, hashtags) and industry specific words etc <br/>
* Word/Lexicon Normalization - Tokenization, Lemmatization, Stemming <br/>
* Word/Object Standardization - Regular Expression, Lookup Tables <br/>
* Stopwords should be used moderately when using LSTMs 
##### Noise Removal
Any piece of text which is not relevant to the context of the data and the end-output can be specified as the noise <br/>
* Spelling Correction
* Removing Stop Words: Stop words already present in NLTK package, can add new words to the existing stop words <br/>
* Removal of irrelevant characters such as any non-alphanumeric chars, spaces, punctuation, Hashwords, @ twitter mentions, urls etc <br/>
* Removing text file headers, footers
* Expand contractions
* Convert all chars to lowercase or uppercase
##### Lexicon Normalization - Stemming
Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word <br/>
* Porter stemming algorithm - https://tartarus.org/martin/PorterStemmer/
* Lancaster stemming algorithm - 
* Snowball stemming algorithm - http://snowball.tartarus.org/
##### Lexicon Normalization - Lemmatizing
Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words) and morphological analysis (word structure and grammar relations) <br/>
But how is this different than Python stemming? While stemming can create words that do not actually exist, Python lemmatization will only ever result in words that do
##### Text Normalization
* Dictionary mappings
* Statistical Machine Translation (SMT)
* Spelling correction

##### Part-Of-Speech Tagging and POS Tagger

![](https://cdn-images-1.medium.com/max/1600/1*VRgrWGLBJJDXsQE72QyI3g.png)

### Feature Extraction
DL models work with numeric vectors. Vectorizing text is the process of transforming text into numeric tensors.  <br/>
https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41 <br/>
https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa <br/>
#### Count Vectorization (Bag of words)
Count Vector is a matrix notation of the dataset in which every row represents a document from the corpus, every column represents a term from the corpus, and every cell represents the frequency count of a particular term in a particular document
#### N-Grams
An n-gram is a contiguous sequence of n items from a given sample of text or speech.
#### TF-IDF
TF-IDF score represents the relative importance of a term in the document and the entire corpus. TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears <br/>
  TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document) <br/>
  IDF(t) = log_e(Total number of documents / Number of documents with term t in it) <br/>
TF-IDF Vectors can be generated at different levels of input tokens (words, characters, n-grams) <br/>
* Word Level TF-IDF : Matrix representing tf-idf scores of every term in different documents
* N-gram Level TF-IDF : N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams
* Character Level TF-IDF : Matrix representing tf-idf scores of character level n-grams in the corpus
#### Co-Occurence Matrix
Similar words tend to occur together and will have similar context <br/>
Co-occurrence – For a given corpus, the co-occurrence of a pair of words say w1 and w2 is the number of times they have appeared together in a Context Window <br/>
Context Window – Context window is specified by a number and the direction. So what does a context window of 2 (around) means? Let us see an example below <br/>
#### Topic Modeling - LDA
https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
Topic models are extremely useful in summarizing large corpus of text documents to extract and depict key concepts. They are also useful in extracting features from text data that capture latent patterns in the data. <br/>
Latent Dirichlet Allocation is the most popular topic modeling technique. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place. <br/>
LDA is a matrix factorization technique. In vector space, any corpus (collection of documents) can be represented as a document-term (term = words in the documents) matrix. LDA converts this Document-Term Matrix into two lower dimensional matrices – M1 and M2. <br/>
M1 is a document-topics matrix and M2 is a topic–terms matrix with dimensions (N,K) and (K,M) respectively, where N is the number of documents, K is the number of topics and M is the vocabulary size. <br/>
* Alpha and Beta Hyperparameters – alpha represents document-topic density and Beta represents topic-word density. Higher the value of alpha, documents are composed of more topics and lower the value of alpha, documents contain fewer topics. On the other hand, higher the beta, topics are composed of a large number of words in the corpus, and with the lower value of beta, they are composed of few words. <br/>
* Number of Topics – Number of topics to be extracted from the corpus. Researchers have developed approaches to obtain an optimal number of topics by using Kullback Leibler Divergence Score.
* Number of Topic Terms – Number of terms composed in a single topic. It is generally decided according to the requirement. If the problem statement talks about extracting themes or concepts, it is recommended to choose a higher number, if problem statement talks about extracting features or terms, a low number is recommended.
* Number of Iterations / passes – Maximum number of iterations allowed to LDA algorithm for convergence.

#### Document Clustering with Similarity Features 
Built on top of Count Vectorized Matrix or TF-IDF matrix
Hierarchical clustering

#### Named Entity Recognition
Entities are defined as the most important chunks of a sentence – noun phrases, verb phrases or both. Entity Detection algorithms are generally ensemble models of rule based parsing, dictionary lookups, pos tagging and dependency parsing. The applicability of entity detection can be seen in the automated chat bots, content analyzers and consumer insights.<br/>
A typical NER model consists of three blocks:<br/>
* Noun phrase identification: This step deals with extracting all the noun phrases from a text using dependency parsing and part of speech tagging.<br/>
* Phrase classification: This is the classification step in which all the extracted noun phrases are classified into respective categories (locations, names etc). Google Maps API provides a good path to disambiguate locations, Then, the open databases from dbpedia, wikipedia can be used to identify person names or company names. Apart from this, one can curate the lookup tables and dictionaries by combining information from different sources.<br/>
* Entity disambiguation: Sometimes it is possible that entities are misclassified, hence creating a validation layer on top of the results is useful. Use of knowledge graphs can be exploited for this purposes. The popular knowledge graphs are – Google Knowledge Graph, IBM Watson and Wikipedia. <br/>

#### Diff between word2vec, Glove, ELMo, BERT
![](https://qph.fs.quoracdn.net/main-qimg-b46d83b2eee2d5875f469b22a494db6e)
* Word2vec and Glove are context independent. One word has only 1 vector. We cam just use vectors from the words to apply it to our new data
* ELMo & BERT are context dependent. So, instead of just the vectors for the words we need the Training model as well



### Word Embeddings
* Word2Vec - uses CBOW or Skip-gram. To make CBOW or skip-gram algorithm computationally more efficient, tricks like negative sampling, softmax or Hierarchical Softmax loss functions are used
* GloVe
* FastText - Extension of Word2Vec proposed by Facebook. 

### Pre-Trained Language Models
* ULMFit
* ELMo - 
* Transformer
* BERT - Uses Attention Transformers


#### Word2Vec
Word embeddings are learnt by starting with random word vectors and then they get updated in the same way the weights of neural network do to better learn the mapping between input x and output label y
https://code.google.com/archive/p/word2vec/ <br/>
Use Negative sampling to train word2vec efficiently http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/ <br/>
Word2Vec, GloVe <br/> 
* Unsupervised learning algorithm and it works by predicting its context words by applying a two-layer neural network. Word2vec/Gove gives only one numeric representation for a word regardless of the diff meanings that they may have.
* Converts words to a meaningful numeric value <br/>
* Based on the assumption that the meaning of a word can be inferred by the company it keeps
* word2vec representation is created using 2 algorithms: Continuous Bag-of-Words model (CBOW) and the Skip-Gram model. <br/>
Algorithmically, these models are similar, except that CBOW predicts target words (e.g. 'mat') from source context words ('the cat sits on the') i.e 'predicting the word given its context' , while the skip-gram does the inverse and predicts source context-words from the target words i.e 'predicting the context given a word'. <br/>
* word embedding dimension is determined by computing (in an unsupervised manner) the accuracy of the prediction
* Word2vec can be used in recommendations. If a person has listened to song A, try searching for songs that other persons listened to before of after listening to song A and recommend the songs that are close to A to the person. A user’s listening queue as a sentence, with each word in that sentence being a song that the user has listened to. So then, training the Word2vec model on those sentences essentially means that for each song the user has listened to in the past, we’re using the songs they have listened to before and after to teach our model that those songs somehow belong to the same context. 
* Problems With CBoW/Skip-gram
  * Firstly, for each training sample, only the weights corresponding to the target word might get a significant update. While training a neural network model, in each back-propagation pass we try to update all the weights in the hidden layer. The weight corresponding to non-target words would receive a marginal or no change at all, i.e. in each pass we only make very sparse updates.
  * Secondly, for every training sample, the calculation of the final probabilities using the softmax is quite an expensive operation as it involves a summation of scores over all the words in our vocabulary for normalizing.
  * So for each training sample, we are performing an expensive operation to calculate the probability for words whose weight might not even be updated or be updated so marginally that it is not worth the extra overhead.
  * To overcome these two problems, instead of brute forcing our way to create our training samples, we try to reduce the number of weights updated for each training sample. 
* Negative Sampling: it suggests that instead of backpropagating all the 0s in the correct output vector (for a vocab size of 10mill there are 10mill minus 1 zeros) we just backpropagate a few of them (say 14). 
  * Negative sampling allows us to only modify a small percentage of the weights, rather than all of them for each training sample. We do this by slightly modifying our problem. Instead of trying to predict the probability of being a nearby word for all the words in the vocabulary, we try to predict the probability that our training sample words are neighbors or not. Referring to our previous example of (orange, juice), we don’t try to predict the probability for juice to be a nearby word i.e P(juice|orange), we try to predict if (orange, juice) are nearby words or not by calculating P(1|<orange, juice>).
  * So instead of having one giant softmax — classifying among 10,000 classes, we have now turned it into 10,000 binary classification problem.
  * We further simplify the problem by randomly selecting a small number of “negative” words k(a hyper-parameter, let’s say 5) to update the weights for. (In this context, a “negative” word is one for which we want the network to output a 0).
  * For our training sample (orange, juice), we will take five words, say apple, dinner, dog, chair, house and use them as negative samples. For this particular iteration we will only calculate the probabilities for juice, apple, dinner, dog, chair, house. Hence, the loss will only be propagated back for them and therefore only the weights corresponding to them will be updated.
* Hierarchical Softmax: Calculating the softmax for a vocab of 10mill is very time and computation intensive. Hierarchical Softmax suggests a faster way of computing it using Huffman trees

Applications
* Music recommendations - https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484
* Find Similar Quora Questions - https://towardsdatascience.com/finding-similar-quora-questions-with-word2vec-and-xgboost-1a19ad272c0d
* 

Training of Word2vec on Wiki Corpus https://medium.com/@maxminicherrycc/word2vec-how-to-train-and-update-it-4eed4260cf75 <br/>
Train and Update Word2vec https://medium.com/@maxminicherrycc/word2vec-how-to-train-and-update-it-4eed4260cf75 <br/>
Word2vec to Song2vec https://medium.com/@weiqi_tong/from-word2vec-to-song2vec-an-embedding-experimentation-9215279c9d7a <br/>
Detailed demons of CBOW, skip-gram, Feature engineering using word2vec(end of the article) https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa <br/>
Google's trained Word2Vec http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/ <br/>
After importing google's model  https://github.com/chrisjmccormick/inspect_word2vec/blob/master/inspect_google_word2vec.py <br/>
 https://medium.com/swlh/playing-with-word-vectors-308ab2faa519   <br/>
 https://github.com/mkonicek/nlp <br/>
FastText's trained word vectors model https://fasttext.cc/docs/en/english-vectors.html <br/>
Detained Neural network representation https://iksinc.online/tag/continuous-bag-of-words-cbow/ <br/>
https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/ <br/>
https://www.datascience.com/resources/notebooks/word-embeddings-in-python <br/>


#### GloVe - Global Vectors for Word Representation
https://medium.com/@SAPCAI/glove-and-fasttext-two-popular-word-vector-models-in-nlp-c9dc051a3b0  <br/>
https://towardsdatascience.com/comparing-word-embeddings-c2efd2455fe3 <br/>
https://towardsdatascience.com/representing-text-in-natural-language-processing-1eead30e57d8 </br>
* Both CBOW and Skip-Grams are “predictive” models, in that they only take local contexts into account. Word2Vec does not take advantage of global context (Word co-occurence). GloVe embeddings by contrast leverage the same intuition behind the co-occuring matrix used distributional embeddings, but uses neural methods to decompose the co-occurrence matrix into more expressive and dense word vectors. While GloVe vectors are faster to train, neither GloVe or Word2Vec has been shown to provide definitively better results rather they should both be evaluated for a given dataset
* GloVe captures both global statistics (Global matrix factorizations when applied to term frequency matrices are called Lastent Semantic Analysis) and local statistics (CBOW & Skip-gram)of a corpus, in order to come up with word vectors.
* GloVe optimizes the embeddings directly so that the dot product of two word vectors equals the log of the number of times the two words will occur near each other (within a 2-words window, for example). This forces the embeddings vectors to encode the frequency distribution of which words occur near them.
* GloVe brings up more infrequent similar words that the other models, which becomes quite overwhelming in the tail.
* GloVe (glove.42B.300d): 300-dimensional vectors trained on the 42B token Common Crawl corpus

#### Continuous Bag-of-Words
https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html
Running CBOW is computationally expensive and works better if trained using a GPU. Guy in the above article used AWS p2.x instance with a Tesla K80 GPU and it took me close to 1.5 hours for just 5 epochs! <br/>
* CBOW is faster and has better representations for more frequent words.

#### Skip-Gram Model
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
* Skip Gram works well with small amount of data and is found to represent rare words well.

#### FastText 
fastText WIKI (wiki-news-300d-1M): 300-dimensional vectors trained on the 16B token Wikipedia 2017 dump
* Similar to Word2vec, fastText also supports training CBOW or Skip-gram models using Negative sampling, softmax or Hierarchical softmax loss functions
* FastText, builds on Word2Vec by learning vector representations for each word and the n-grams found within each word. The values of the representations are then averaged into one vector at each training step. While this adds a lot of additional computation to training it enables word embeddings to encode sub-word information. FastText vectors have been shown to be more accurate than Word2Vec vectors by a number of different measures
* For instance, the tri-grams for the word apple is app, ppl, and ple (ignoring the starting and ending of boundaries of words). The word embedding vector for apple will be the sum of all these n-grams. After training the Neural Network, we will have word embeddings for all the n-grams given the training dataset. Rare words can now be properly represented since it is highly likely that some of their n-grams also appears in other words. 
* Word2vec and GloVe both fail to provide any vector representation for words that are not in the model dictionary. This is a huge advantage of this method.


#### Sentence Vectors
https://medium.com/explorations-in-language-and-learning/how-to-obtain-sentence-vectors-2a6d88bd3c8b
* Paragraph Vectors - a sentence vector can be learned simply by assigning an index to each sentence, and then treating the index like any other word.
* Skip-thoughts
* FactSent
* Sequential Denoising Autoencoders (SDAE)

#### Doc2Vec
https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e <br/>
https://medium.com/explorations-in-language-and-learning/how-to-obtain-sentence-vectors-2a6d88bd3c8b <br/>

#### ELMo - Embeddings from Language Models
https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/ </br>
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/03/output_YyJc8E.gif)
* ELMo and BERT can generate diff word embeddings for a word that captures the context of a word - that is its position in a sentence
* ELMo uses LSTMs
* ELMo is a charcater based model using character convolutions and can handle out of vocab words, but learnt representations are at word level
* Model is trained to predict the next word given a sequence words
* ELMo word vectors are computed on top of a two-layer bidirectional language model (biLM). This biML model has two layers stacked together. Each layer has 2 passes — forward pass and backward pass:
  * The architecture above uses a character-level convolutional neural network (CNN) to represent words of a text string into raw word vectors
  * These raw word vectors act as inputs to the first layer of biLM
  * The forward pass contains information about a certain word and the context (other words) before that word
  * The backward pass contains information about the word and the context after it
  * This pair of information, from the forward and backward pass, forms the intermediate word vectors
  * These intermediate word vectors are fed into the next layer of biLM
  * The final representation (ELMo) is the weighted sum of the raw word vectors and the 2 intermediate word vectors </br>
As the input to the biLM is computed from characters rather than words, it captures the inner structure of the word. For example, the biLM will be able to figure out that terms like beauty and beautiful are related at some level without even looking at the context they often appear in.
* Traditional word embeddings such as word2vec and GLoVe, the ELMo vector assigned to a token or word is actually a function of the entire sentence containing that word. Therefore, the same word can have different word vectors under different contexts.
  * Word 'read' can be used as verb in present as well as past tense. Traditional word embeddings come up with same vector for the word 'read' in both the sentences. ELMo word vectors successfully address this issue.
* ELMo word representations take the entire input sentence into equation for calculating the word embeddings. Hence, the term “read” would have different ELMo vectors under different context.
  
#### BERT (Bidirectional Encoder Representations from Transformers)
* BERT uses Transformer - an attention based model with positional encodings to represent word positions
* BERT is the first unsupervised, deeply bidirectional system for pre-training NLP. It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks.
* BERT produces 768 dimensional vector for each word based on their context. A single word with multiple meanings will have different vectors.
* https://www.coursera.org/learn/ai-for-medical-treatment/lecture/CotPU/handling-words-with-multiple-meanings </br> LEARNING PROCESS: Let's see how BERT learns these contextualized word representations. Words from a passage of text are input into a BERT model. Then one of the tokens in the passage is masked with a special MASK token. The model is trained to predict what the mask was. An extra layer is added where the output is the probabilities of the missing word being every single word in the vocabulary. Here we can see that the true missing word debate gets a probability output of the model of 0.7. In the process of learning to correctly predict the masked word, the model learns word representations here in blue. BioBERT uses passages from medical papers to learn these word representations. The advantage of this is that the words that BioBERT is thus able to learn, are words used in the context of medicine.
* https://www.coursera.org/learn/ai-for-medical-treatment/lecture/L34sl/define-the-answer-in-a-text </br> QUESTION ANSWERING CONTEXT: For the question answering task, the inputs will be the question and the passage, and the output will be the answer to the question which is a segment of the passage. The task for the model is to be able to determine whether each word in the passage is one of the start or the end of an answer to a question. Here's how the model learns to determine whether a word is likely to be the start or the end word to an answer. The model learns two vectors, S and E for each of the word representations, for each of the words in the passage. The word representation is multiplied by S to get a single number, which is the start score for that word. The higher the start score, the more likely it is to be the start of the answer. Similarly, for each of the word representations, the word representation is multiplied by the vector E to get another scalar number, which is the end score. The higher the end score, the more likely the word is to be the end of an answer. Using the start and end scores for each of the words, we can find out what the most likely answer is. We do this by computing a grid of words. In this grid, we enter in this start score plus the end score in each of the cells, the start score coming from the rows, and the end score coming from the column. For instance, to compute the score that the answer starts with blood and ends with glucose, we would fill in this cell and we would get these start score from blood, that's 0.1, and add it to the end score from glucose, which is here again 0.1, to get us a score in the cell of 0.2. We can thus compute the scores for all entries in the grid. We can force the end word to appear no earlier than the start word by only computing the scores in this upper triangular region. Remember that if we want here, then we're saying that the start would be after the end, which is not possible. The model thus outputs the start and end word corresponding with the highest score here. The highest score here is 8.2, which has a start with 'reduce' and an end with levels. So we have the model output 'reduce' as the start of the answer at token 11, and 'levels' as the end of the answer at token 14. The model learns the vectors S and E and updates its word representations based on being shown many of these question, passage, and answer triplets. Typically, the model is first shown natural question and answers in English in the general domain using datasets like SQuAD and then fine tune on medical datasets like BioASQ.


* BERT represents inputs as subwords and learns embeddings from subwords
* 


#### GPT


#### XLNet
* 

#### ULMfit
*  Universal Language Model Fine-tuning (ULMFiT), an effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model.
* Small dataset is sufficient in Transfer learning training by using ULMFit
* ULMFiT involves 3 major stages: LM pre-training, LM fine-tuning and Classifier fine-tuning. The method is universal in the sense that it meets these practical criteria: https://towardsdatascience.com/understanding-language-modelling-nlp-part-1-ulmfit-b557a63a672b
  * It works across tasks varying in document size, number, and label type.
  * It uses a single architecture and training process.
  * It requires no custom feature engineering or pre-processing.
  * It does not require additional in-domain documents or labels.
* LM pre-training: The LM is trained on a general-domain corpus to capture general features of the language in different layers. We pre-train LM on a large general-domain corpus and fine-tune it on the target task using novel techniques. So, authors have used Wikitext-103 a dataset of 28k preprocessed articles consisting of 103Million words. In general, the dataset should be so huge that the LM learns all the properties of the langauge. This is the most expensive in terms of compute resources and time too. Hence we do this just once.
* LM fine-tuning: In almost all the cases the target task dataset will have a different distribution w.r.t. the general domain corpus. In this stage, we fine-tune the model on the target task dataset to learn its distributions by using discriminative fine-tuning and slanted triangular learning rates. STLR has been used to achieve state-of-the-art results in CV (Python: language_model_learner and fit base on the optimal learning rate)
  * As different layers grasp different information, author suggest to fine-tune each layer to a different extent.
  * In Stochastic Gradient Descent we update θ at each time step t.
  * In discriminative fine-tuning, we use θ1, θ2,… θL instead of singel θ value for respective L layers.
  * In STLR, authors suggest to increase the learning rate linearly and decay it in the following manner.
* Classifier fine-tuning: Fine-tuning being the most vital state of transfer learning needs to be done with maximum care. Because an aggressively done fine-tuning could over-fit our model and vice versa could make our model under-fit. Authors have suggested Gradual unfreezing approach to deal with this major issue. We start with unfreezing only the last layer as its contains the most general knowledge. After fine-tuning unfrozen layers for one epoch, we go for next lower layer and repeat till we complete all layers until convergence at the last iteration.   

### Visualizing Features
In order to see whether our embeddings/features are capturing information that is relevant to our problem , it is a good idea to visualize them and see if the classes look well separated. Since vocabularies are usually very large and visualizing data in 20,000 dimensions is impossible, techniques like PCA will help project the data down to two dimensions. And is then plotted. 2 features on X & Y-axis and target is colour coded

##### T-SNE T-Distributed Stochastic Neighborhood Embedding
https://medium.com/@sourajit16.02.93/tsne-t-distributed-stochastic-neighborhood-embedding-state-of-the-art-c2b4b875b7da
* Dimensionality reduction method used for the visualization of very high dimensional data
* PCA (performs linear mapping) tries to maximize variance towards Principal components, tries to maintain GLOBAL structure of the data
* TSNE, unlike PCA, preserves the local structures (also) of the data points while converting from higher to lower dimensions.
* It is a probablistic (non-linear) technique </br>
https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/ </br>
https://www.datacamp.com/community/tutorials/introduction-t-sne </br>
![](https://cdn-images-1.medium.com/max/2400/1*Pb9EpsHGF3umWEWPswU4rQ.png)

### Evaluation 
##### BLEU (Bilingual Evaluation Understudy)
It is mostly used to measure the quality of machine translation with respect to the human translation. It uses a modified form of precision metric. <br/>
Steps to compute BLEU score: <br/>
1. Convert the sentence into unigrams, bigrams, trigrams, and 4-grams
2. Compute precision for n-grams of size 1 to 4
3. Take the exponential of the weighted average of all those precision values 
4. Multiply it with brevity penalty (will explain later)
##### Banana Test
https://towardsdatascience.com/introducing-the-banana-test-for-near-perfect-text-classification-models-ee333abfa31a
* 


Example NLP Pipeline
1. retrieve free-text record
2. sentence fragmentation
3. tokenization (word separation)
4. part of speech tagging (labeling nouns, adv, adj, etc)
5. noun phrase detection (NP)
6. concept / named entity recognition (umls,snomed, etc)
7. extraction of semantic relations
8. output structured results 

NLP Pipeline for Low level processing
Tokenization--> Sentence Segmentation --> Part of speech Tagging --> Stemming

Higher level tasks - coreference resolution, NER
