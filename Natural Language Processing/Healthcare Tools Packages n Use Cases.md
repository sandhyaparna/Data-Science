### Tools n Packages
* Valx https://github.com/Tony-Hao/Valx </br>
Demo http://columbiaelixr.appspot.com/valx </br>
* Disease Extraction - https://ii-public1.nlm.nih.gov/metamaplite/rest https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5884681/
* SciSpacy https://github.com/allenai/scispacy </br>
  * https://colab.research.google.com/drive/1sSk6PDqosn7ozSZUZj4LW5HoRypmNKug#scrollTo=q85ziDCOLU-t
* UMLS 
  * UMLS Terminology Services https://uts.nlm.nih.gov/metathesaurus.html;jsessionid=9E3CEB9102EB724EA13204C883CAB7F6#C0005823;0;2;CUI;2019AA;EXACT_MATCH;CUI;*;
  * UMLS csv files https://github.com/maastroclinic/umls-csv-concept-integration
  * 
* ClarityNLP (Clarity.NamedEntityRecognition) https://buildmedia.readthedocs.org/media/pdf/claritynlp/latest/claritynlp.pdf </br>
* Apache cTAKES https://clinfowiki.org/wiki/index.php/CTAKES </br>
* I-MAGIC tool for SNOMED CT to ICD-10 Map https://www.nlm.nih.gov/research/umls/mapping_projects/snomedct_to_icd10cm.html#:~:text=It%20is%20designed%20for%20use,for%20reimbursement%20and%20statistical%20purposes. </br>
https://imagic.nlm.nih.gov/imagic/code/map?v=5&js=true&pabout=&pinstructions=&p=e2551282zd558769e10&p.e2551282zd558769e10.s=709044004&pat=My+Patient+%28modified%29&pat.init=My+Patient+%28modified%29&q.f=&q.dob=&plist=Back+to+Problem+List
* CliNER https://github.com/text-machine-lab/CliNER </br>
Demo http://text-machine.cs.uml.edu/cliner/demo/cgi-bin/cliner_demo.cgi/ </br>
* medaCy - Similar to Spacy https://github.com/NLPatVCU/medaCy </br>
* Stanford NER https://nlp.stanford.edu/software/CRF-NER.html </br>
Demo http://nlp.stanford.edu:8080/ner/process </br>
Demo http://corenlp.run/ </br>
* PyUMLS </br>
* Clinspell: Clinical Spell Correction https://github.com/clips/clinspell </br>
* Unsupervised Extraction of Diagnosis Codes from EMRs Using Knowledge-Based and Extractive Text Summarization Techniques https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5524149/

### Relation Extractor
https://arxiv.org/abs/1901.08746 </br>
http://nlpprogress.com/english/relationship_extraction.html </br>
https://sflscientific.com/data-science-blog/2015/12/11/natural-language-processing-information-extraction </br>
https://towardsdatascience.com/conditional-random-field-tutorial-in-pytorch-ca0d04499463 </br>
http://sameersingh.org/courses/statnlp/wi17/slides/lecture-0223-relation-extraction.pdf </br>
http://deepdive.stanford.edu/relation_extraction </br>
https://courses.cs.washington.edu/courses/cse517/13wi/slides/cse517wi13-RelationExtraction.pdf </br>
https://github.com/UKPLab/emnlp2017-relation-extraction </br>
https://www.nltk.org/book/ch07.html </br>
http://www.nltk.org/howto/relextract.html </br>
 </br>
 </br>
 </br>


### Companies ###
* Linguamatics
* Regenstrief Institute's nDepth
* cTakes - Requires UMLS license
https://medium.com/@felix_chan/install-apache-ctakes-924c40967ce2 </br>
https://cwiki.apache.org/confluence/display/CTAKES/cTAKES+3.2+User+Install+Guide </br>
https://www.youtube.com/watch?v=4aOnafv-NQs </br>
* i2b2 - 


### Use Cases 
* Extract & associate numerical attributes and values from unstructured EMR data
  * Extract Attributes and values using Valx, CliNER or Stanford NER
  * Associates values to their respective attributes </br>
https://arxiv.org/ftp/arxiv/papers/1602/1602.00269.pdf </br>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5543347/ </br>
https://pdfs.semanticscholar.org/d3c9/8c90847eb1739819c024eb39a8095fe1ee32.pdf </br>

* Diagnosis code extraction 
  * Hierarchy of ICD codes (Sequential trees LSTM)
  * Diagnosis descriptions (Adversarial learning)
  * Rank ICD codes (ADMM Isotonic constraints)
  * Many-to-many mapping between diagnosis descriptions & ICD codes </br>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5524149/ </br>
https://www.aclweb.org/anthology/P18-1098 </br>
https://pdfs.semanticscholar.org/65f0/a9a1f626bd6c3108a8f9eb6c70cad89ce41e.pdf </br>
https://arxiv.org/pdf/1711.04075.pdf </br>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5524149/ </br>

##### Information Extraction (IE)
* Mono and multilingual  
* NER
* Text classification
* Acronym normalization
* Form filling

##### Information Management
* eHealth data visualization
* Medical reports management

##### Information retrieval (IR)
* Mono- and multilingual IR
* Session-based IR


### Sample Data
* NCT00021788, NCT00211536, NCT00297583
* Her vital signs the following day, she had heart rate of 66, blood pressure 120/63, respiratory rate 14, 100% on 5 liters nasal cannula O2 saturation
* Dr. Nutritious </br>
  Medical Nutrition Therapy for Hyperlipidemia </br>
  Referral from: Julie Tester, RD, LD, CNSD </br>
  Phone contact: (555) 555-1212 </br>
  Height: 144 cm   Current Weight: 45 kg   Date of current weight: 02-29-2001   Admit Weight:  53 kg   BMI: 18 kg/m2 </br>
  Diet: General </br>
  Daily Calorie needs (kcals): 1500 calories, assessed as HB + 20% for activity. </br>
  Daily Protein needs: 40 grams,  assessed as 1.0 g/kg. </br>
  Pt has been on a 3-day calorie count and has had an average intake of 1100 calories.  She was instructed to drink 2-3 cans of liquid supplement to help promote weight gain.  She agrees with the plan and has my number for further assessment. May want a Resting Metabolic Rate as well. She takes an aspirin a day for knee pain. </br>
* https://github.com/sandhyaparna/Python-DataScience-CookBook/blob/master/Natural%20Language%20Processing/Data/dates.txt
* https://github.com/sandhyaparna/Python-DataScience-CookBook/blob/master/Natural%20Language%20Processing/Data/example%20data%20diabetes_Type%201.csv
* - Her vital signs: heart rate of 66, blood pressure 120/63, respiratory rate 14, 100% on 5 liters nasal cannula O2 saturation#  - HbA1c less than or equal to 11.0%#

### POCS for ICD 10 extraction
* MIMIC database (Download from https://physionet.org/content/mimiciii/1.4/) can be used for Unsupervised Extraction of Diagnosis Codes (ICD10 codes) from Electronic medical Records
* Overview: ICD9 is used for diagnosis coding of MIMIC data. ICD 9 has around 14k diagnosis codes where as ICD 10 has 69k diagnosis codes.
* UMLS (Unified Medical Language System): UMLS is a large domain expert driven aggregation of over 160 biomedical terminologies and standards. It functions as a comprehensive knowledge base and facilitates interoperability between information systems that deal with biomedical terms. It has has three main components: Metathesaurus, Semantic Network, and SPECIALIST lexicon. The Metathesaurus has terms and codes, henceforth called concepts, from different terminologies. Biomedical terms from different vocabularies that are deemed synonymous by domain experts are mapped to the same Concept Unique Identifier (CUI) in the Metathesaurus. The semantic network acts as a typing system that is organized as a hierarchy with 133 semantic types such as disease or syndrome, pharmacologic substance, or diagnostic procedure. It also captures 54 important relationships (or relation types) between biomedical entities in the form of a relationship hierarchy with relationships such as treats, causes, and indicates. The Metathesaurus currently has about 2.8 million concepts with more than 12 million relations connecting these concepts. The SPECIALIST lexicon is useful for lexical processing and variant generation of different biomedical terms.
Links: https://www.nlm.nih.gov/research/umls/quickstart.html
* SNOMED CT (Systematized Nomenclature of Medicine -- Clinical Terms): It is part of UMLS Metathesaurus. The great majority of the differences between the concept structures of SNOMED CT and the UMLS Metathesaurus are due to a few differences in their concept models. In several cases, SNOMED CT's description logic defines as distinct concepts what the Metathesaurus sees as different perspectives on a single concept.
Link: https://www.nlm.nih.gov/research/umls/Snomed/snomed_represented.html#:~:text=In%20SNOMED%20CT%20the%20RelationshipType,116680003')%20to%20enhance%20readability.
* I-MAGIC (Interactive Map-Assisted Generation of ICD Codes) Algorithm utilizes the SNOMED CT to ICD-10-CM Map in a real-time, interactive manner to generate ICD-10-CM codes but 





