### Tools n Packages
* UMLS
* Apache cTAKES https://clinfowiki.org/wiki/index.php/CTAKES </br>
* I-MAGIC tool for SNOMED CT to ICD-10 Map https://imagic.nlm.nih.gov/imagic2/code/map?v=5&js=true&pabout=&pinstructions=&init-params=&pat=My+Patient+%28modified%29&pat.init=My+Patient+%28modified%29&q.f=&q.dob=&p=2a0333c3zd3542e10&p.2a0333c3zd3542e10.e=Respiratory+distress+secondary+to+transient+tachypnea+of+the+newborn&p.2a0333c3zd3542e10.o=Respiratory+distress+secondary+to+transient+tachypnea+of+the+newborn&pdone=Get+ICD+Codes&qadd= </br>
* CliNER https://github.com/text-machine-lab/CliNER </br>
Demo http://text-machine.cs.uml.edu/cliner/demo/cgi-bin/cliner_demo.cgi/ </br>
* medaCy - Similar to Spacy https://github.com/NLPatVCU/medaCy </br>
* Valx https://github.com/Tony-Hao/Valx </br>
Demo http://columbiaelixr.appspot.com/valx </br>
* Stanford NER https://nlp.stanford.edu/software/CRF-NER.html </br>
Demo http://nlp.stanford.edu:8080/ner/process
* PyUMLS

### Companies ###
* Linguamatics
* Regenstrief Institute's nDepth
* cTakes - Requires UMLS license
https://medium.com/@felix_chan/install-apache-ctakes-924c40967ce2 </br>
https://cwiki.apache.org/confluence/display/CTAKES/cTAKES+3.2+User+Install+Guide </br>
https://www.youtube.com/watch?v=4aOnafv-NQs </br>


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








