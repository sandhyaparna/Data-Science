### Automatic label extraction for Medical Imaging (Unsupervised):
* Two steps: In step one, we will find whether an observation is mentioned. In step two, we will classify whether the observation is present or absent. 
* Uses SNOMED CT - contains Synonyms, Is-a relationships
* Step One: In the eg note "Heart size is normal and lungs are clear. No edema or Pneumonia. No effusion". This note has the mention of pneumonia. However, what this won't be able to catch are other words which may have the same meaning in this context as pneumonia. For example, infection can be synonymous with pneumonia. So instead of searching for pneumonia, we may list out the words which are synonyms of pneumonia, and search the report for any of those words, that way, catch infection being mentioned.  
  * Let's see this for another observation. Lesion, which includes changes involving any tissue or organ due to disease or injury. Once again, for lesion, we might start with the word lesion. We can try to get all the ways we can say lesion by asking a radiologist. Radiologists might come back with a list of all words that might be synonymous with lesion. Of course, this requires knowledge in radiology or access to a radiologists for each such category. 
  * Another option is to use a terminology, also called a thesaurus or a vocabulary. These are an organization collection of medical terms to provide codes, definitions, and synonyms. Example of such a terminology is SNOMED CT, which consists of 300,000 concepts. An example concept might be common cold, which would contain a concept number and would contain synonyms for the common cold. It's thus possible to find mentions of pneumonia by looking up the concept of pneumonia using SNOMED CT or other terminology and retrieving its synonyms to search the report for. 
  * Let's see another way in which terminologies can help us. Let's say we want to find mentions of lung disease. As before, we can search the report for lung disease or its synonyms using a terminology. However, this might return no results because infection is not a direct synonym of lung disease. Now this report tells us that there is a mention of pneumonia, which is a type of lung disease. So we should be able to say yes for mention of lung disease. How do we tackle this challenge? We can go back to terminologies to help us. Terminologies not only contain synonyms for our concept but also contain relationships to other concepts. Here, we can see that common cold has a Is-A relationship with Viral Upper Respiratory Tract Infection. Similarly for pneumonia we can see a hierarchy of relationships. How one specific type of pneumonia is an infectious pneumonia, which is a pneumonia which is a lung disease. 
  * Therefore we can catch mention of lung disease by not only searching for synonyms of lung disease in SNOMED CT, but also its subtypes and their synonyms. Here subtypes would include pneumonia and other concepts which have a Is- A relationship with lung disease. The advantage of this approach, which we can call a rules-based approach of finding mentions of observations, is that we don't need any data for supervised learning. The disadvantage of this approach is that there is a lot of manual work to refine these rules based on what is working and what is not working.
* Step Two: 
  * Regex Rules: No edema, No xxx or edema, without xxx edema, no evidence of edema
  * Dependency Parse Rules
  * Negation classification - Supervised approach. 
  * NegBio
##### Links
* BioC: http://bioc.sourceforge.net/
* NegBio: https://github.com/ncbi-nlp/NegBio ; https://negbio.readthedocs.io/en/latest/index.html

### Evaluate Automatic Labeling
* Using ground growth from experts annotating the Text report or looking at an image for presence or absence of a disease. The advantage of using the ground truth from the report is that it would be more straightforward to improve the system based on looking at the errors made on the report. The advantage of using a ground truth from the image is that it is a more direct evaluation of the quality of the label for the task.















