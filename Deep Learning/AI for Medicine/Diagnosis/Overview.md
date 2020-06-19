### Course Info
* Link - https://www.coursera.org/learn/ai-for-medical-diagnosis/home/week/1
* Data in Images folder - Download tar.gz files seperately https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737
* PreRequisites - CNN, loss functions, Probability, Python
* Use Cases
  * Skin images to identify if a suspecious mole is cancerous or not
  * Opthalmology - Diabetic Rhetinopathy - Back of the eye retina images
  * Histopathology - Scanned microscopic images of tissue called whole slide images - to determine the extent to which a cancer has spread and then use to plan treatment, predict the course of the disease, and the chance of recovery
  * Chest X ray - To detect Pneumonia, lung cancer or Mass can be identified. 

### Medical Imaging
* Data of Chest X rays - https://arxiv.org/abs/1705.02315
* Challenges for training medical images - Class Imbalance; Multitask Challenge; Dataset size
* Class Imbalance - Weighted loss function to train data / Resampling(Undersampling, Oversampling)
* Multitask challenge - Training of the multitask algorithm requires modification of loss function from binary tasks to multitask setting. One label for each task. Sum of individual losses. For weighted, fraction of Wp & Wn are different for different binary losses. https://www.coursera.org/learn/ai-for-medical-diagnosis/lecture/VNkO2/multi-task-loss-dataset-size-and-cnn-architectures
* Dataset size Challenge - Tranfer learning + Data Augmentation
  * Transfer Learning - Pre-training (copy/transfer learned featured) and then fine-tuning.  Early layers of the network usually captures low-level image features that are broably generalizable, while the later layers capture details that are more high-level or more specific to a task. So when fine-tuning the network instead of fine-tuning all features we've transferred, we can freeze the features learned by the shallow layers and just fine-tune the deeper layers. In practice, two of the most common design choices are one, to fine-tune all of the layers, and two, only fine-tune the later or the last layer and not fine-tune the earlier layers. This approach of pre-training and fine-tuning, is also called transfer learning and is an effective way to tackle the small dataset size challenge.
  * Trick the network into thinking that we have more training examples - (Data Augmentation)Apply transformations like rotate it, flip left to right and vice-versa,translate it sideways or zoom or change brightness/contrast. Variations should reflect real world scenarios and see if transformation preserves the label or not
  
  
 
 ### Modeling
 * Loss function measures the error between our output probability and the desired label
   * Weighted loss function is to be used. Modified Log loss = ( (Num Negative/No Total) * Σ(-logloss of probability of Y=1 if y=1) ) + ( (Num Positive/No Total) * Σ(-logloss of probability of Y=0 if y=0) )
 * Densenet - https://arxiv.org/pdf/1608.06993.pdf. 
   * Early layers of the network usually captures low-level image features that are broably generalizable, while the later layers capture details that are more high-level or more specific to a task. So when fine-tuning the network instead of fine-tuning all features we've transferred, we can freeze the features learned by the shallow layers and just fine-tune the deeper layers. In practice, two of the most common design choices are one, to fine-tune all of the layers, and two, only fine-tune the later or the last layer and not fine-tune the earlier layers. This approach of pre-training and fine-tuning, is also called transfer learning and is an effective way to tackle the small dataset size challenge.
   
 
### Healthcare Data splitting challenges
  * Patient Overlap - When patient comes twice and have 2 xrays and wears a necklace both the times, but we feed one of the xray into Train and other into Test, there is a high possibility of memorization of unique aspects like necklace in this case and makes the prediction similar to label in Train set. Make sure that a patients xrays are all in either Train set or Test set but not distributed across both Train & Test. SPLIT BY PATIENT.
  * Set sampling - Stratified sampling to get label=1 even when it is rarely occuring. AI for Medical Diagnosis video suggests to take 50% observations of label=1, 50% observations of label=0 for both Test (50% obs with label=1 and 50% obs with label=0) and Validation set (Validation set should reflect the same sampling distribution seen in Test set) and all the remaining observations as Training set. This ensures that the model will have sufficient numbers to get a good estimate of the performance of the model on both non-disease and on disease samples. Order of sampling is Test, Validation & Train https://www.coursera.org/learn/ai-for-medical-diagnosis/lecture/iiAK1/sampling
  * Ground Truth - Consensus voting i.e Humans are asked to determine the outcome and majority voting is used. Use additional medical testing.
 















