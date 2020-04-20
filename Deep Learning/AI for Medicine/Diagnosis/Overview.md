### Course Info
* Link - https://www.coursera.org/learn/ai-for-medical-diagnosis/home/week/1
* PreRequisites - CNN, loss functions, Probability, Python
* Use Cases
  * Skin images to identify if a suspecious mole is cancerous or not
  * Opthalmology - Diabetic Rhetinopathy - Back of the eye retina images
  * Histopathology - Scanned microscopic images of tissue called whole slide images - to determine the extent to which a cancer has spread and then use to plan treatment, predict the course of the disease, and the chance of recovery
  * Chest X ray - To detect Pneumonia, lung cancer or Mass can be identified. 

### Medical Imaging
* Data of Chest X rays - https://arxiv.org/abs/1705.02315
* Challenges for training medical images - Class Imbalance; Multitask Challenge; Dataset size
* For multitask challenge - Training of the multitask algorithm requires modification of loss function from binary tasks to multitask setting. Sum of individual losses. For weighted, fraction of Wp & Wn are different for different binary losses. https://www.coursera.org/learn/ai-for-medical-diagnosis/lecture/VNkO2/multi-task-loss-dataset-size-and-cnn-architectures
* Dataset size Challenge - 
  * Pre-training (copy/transfer learned featured) and then fine-tuning.  Early layers of the network usually captures low-level image features that are broably generalizable, while the later layers capture details that are more high-level or more specific to a task. So when fine-tuning the network instead of fine-tuning all features we've transferred, we can freeze the features learned by the shallow layers and just fine-tune the deeper layers. In practice, two of the most common design choices are one, to fine-tune all of the layers, and two, only fine-tune the later or the last layer and not fine-tune the earlier layers. This approach of pre-training and fine-tuning, is also called transfer learning and is an effective way to tackle the small dataset size challenge.
  * Trick the network into thinking that we have more training examples - (Data Augmentation)Apply transformations like rotate it, flip left to right and vice-versa,translate it sideways or zoom or change brightness/contrast. Variations should reflect real world scenarios and see if transformation preserves the label or not
  
  
 
 ### Modeling
 * Loss function measures the error between our output probability and the desired label
 * Densenet - https://arxiv.org/pdf/1608.06993.pdf. 
   * Early layers of the network usually captures low-level image features that are broably generalizable, while the later layers capture details that are more high-level or more specific to a task. So when fine-tuning the network instead of fine-tuning all features we've transferred, we can freeze the features learned by the shallow layers and just fine-tune the deeper layers. In practice, two of the most common design choices are one, to fine-tune all of the layers, and two, only fine-tune the later or the last layer and not fine-tune the earlier layers. This approach of pre-training and fine-tuning, is also called transfer learning and is an effective way to tackle the small dataset size challenge.
   
 
 
 















