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
#### Word2Vec
https://code.google.com/archive/p/word2vec/ <br/>
Word2Vec, GloVe <br/> 
* Unsupervised learning algorithm and it works by predicting its context words by applying a two-layer neural network. 
* Converts words to a meaningful numeric value <br/>
* Based on the assumption that the meaning of a word can be inferred by the company it keeps
* word2vec representation is created using 2 algorithms: Continuous Bag-of-Words model (CBOW) and the Skip-Gram model. <br/>
Algorithmically, these models are similar, except that CBOW predicts target words (e.g. 'mat') from source context words ('the cat sits on the') i.e 'predicting the word given its context' , while the skip-gram does the inverse and predicts source context-words from the target words i.e 'predicting the context given a word'. <br/>

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

#### Sentence Vectors
https://medium.com/explorations-in-language-and-learning/how-to-obtain-sentence-vectors-2a6d88bd3c8b
* Paragraph Vectors - a sentence vector can be learned simply by assigning an index to each sentence, and then treating the index like any other word.
* Skip-thoughts
* FactSent
* Sequential Denoising Autoencoders (SDAE)



#### Doc2Vec
https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e <br/>
https://medium.com/explorations-in-language-and-learning/how-to-obtain-sentence-vectors-2a6d88bd3c8b <br/>



#### Continuous Bag-of-Words
https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html
Running CBOW is computationally expensive and works better if trained using a GPU. Guy in the above article used AWS p2.x instance with a Tesla K80 GPU and it took me close to 1.5 hours for just 5 epochs! <br/>



#### Skip-Gram Model
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

#### FastText

#### ELMo - Embeddings from Language Models
https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/ </br>
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/03/output_YyJc8E.gif)
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
