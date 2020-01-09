https://github.com/hse-aml/competitive-data-science

# ML Algos
* Linear - Logistic Regression, SVM
  * Good for Sparse High Dimensional data
* Tree Based - Decision Teee, Random Forest, Boosting - Divide & Conquer approach
* KNN - 
* Neural Nets - Frameworks used are Tensorflow, Keras, mxnet, PyTorch, Lasagne

### Data Loading
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline 
from grader import Grader

DATA_FOLDER = '../readonly/final_project_data/' </br>
transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz')) </br>
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv')) </br>
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv')) </br>
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv')) </br>

### 






