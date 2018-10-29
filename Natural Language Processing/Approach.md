#### Links
https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709 <br/>
https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/ <br/>
https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/ <br/>
https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html <br/>
https://www.nltk.org/book/ch06.html <br/>
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html <br/>
 <br/>
https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/ <br/>
https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/ <br/>

### Pre-processing
Tokenization – Process of converting a text into tokens
Tokens – Words or entities present in the text
Text object – A sentence or a phrase or a word or an article
###### Steps
* Noise Removal - Stopwords, Punctuations, URLs or links, social media entities(mentions, hashtags) and industry specific words etc <br/>
* Word/Lexicon Normalization - Tokenization, Lemmatization, Stemming <br/>
* Word/Object Standardization - Regular Expression, Lookup Tables <br/>
###### Noise Removal
Any piece of text which is not relevant to the context of the data and the end-output can be specified as the noise <br/>
* Removing Stop Words: Stop words already present in NLTK package, can add new words to the existing stop words <br/>
* Removal of spaces, punctuation, Hashwords etc <br/>
* Removing text file headers, footers
* Expand contractions
###### Lexicon Normalization - Stemming
Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word <br/>
* Porter stemming algorithm - https://tartarus.org/martin/PorterStemmer/
* Lancaster stemming algorithm - 
* Snowball stemming algorithm - http://snowball.tartarus.org/
###### Lexicon Normalization - Lemmatizing
Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words) and morphological analysis (word structure and grammar relations) <br/>
But how is this different than Python stemming? While stemming can create words that do not actually exist, Python lemmatization will only ever result in words that do
###### Part-Of-Speech Tagging and POS Tagger


### Feature Extraction
###### Count Vectorization
Count Vector is a matrix notation of the dataset in which every row represents a document from the corpus, every column represents a term from the corpus, and every cell represents the frequency count of a particular term in a particular document
###### N-Grams
###### TF-IDF
TF-IDF score represents the relative importance of a term in the document and the entire corpus. TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears <br/>
  TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document) <br/>
  IDF(t) = log_e(Total number of documents / Number of documents with term t in it) <br/>
TF-IDF Vectors can be generated at different levels of input tokens (words, characters, n-grams) <br/>
* Word Level TF-IDF : Matrix representing tf-idf scores of every term in different documents
* N-gram Level TF-IDF : N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams
* Character Level TF-IDF : Matrix representing tf-idf scores of character level n-grams in the corpus
###### Co-Occurence Matrix
Similar words tend to occur together and will have similar context <br/>
Co-occurrence – For a given corpus, the co-occurrence of a pair of words say w1 and w2 is the number of times they have appeared together in a Context Window <br/>
Context Window – Context window is specified by a number and the direction. So what does a context window of 2 (around) means? Let us see an example below <br/>
###### Named Entity Recognition
###### Continuous Bag-of-Words
###### Skip-Gram Model
###### Word Embeddings
###### Part of Speech Tagging
###### Topic Modeling
###### 
###### 
###### 
###### 






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
