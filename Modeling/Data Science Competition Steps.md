https://github.com/hse-aml/competitive-data-science
* Data Pre-processing
  * Numeric - Transformation
  * categorical - Encoding, create interaction vars by concatenating vars 
  * Missing - Replace with a diff value, mean, median; innull_feature, 
* EDA
  * Domain knowledge, Understanding how data was generated, check if data agrees with our domain knowledge
  * Anonymized data - Guess meaning of columns, types of column. Explore feature relations - relations between pairs or find feature groups
  * Visualization - Histograms, plots(index vs values) - plt.plot(x,'.'), stats. Scatter plots - pd.scatter_matrix(df), Correlation plots - df.corr(). Groups - plt.matshow(..); df.mean().plot(style='.'); df.mean().sort_values().plot(style='.')
    * If there is a peak at mean value - it can imply that organizers imputed missing value with mean
    * color coded can be based on 
  * Cleaning - Constant features, Duplicated features - cat features can also be identical but their levels have diff names , Duplicated rows 
    * Duplicated cat feat - for f in cat_features: Df[f] = raintest[f].factorize()
* Validation
* Data-Leakages



# ML Algos
* Linear - Logistic Regression, SVM
  * Good for Sparse High Dimensional data
* Tree Based - Decision Teee, Random Forest, Boosting - Divide & Conquer approach
* KNN - 
* Neural Nets - Frameworks used are Tensorflow, Keras, mxnet, PyTorch, Lasagne

* Data Pre-processing
* Outliers

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

### Pre-processing
* Tree Based models dont depend on scaling
* Non-Tree-Based - Numeric - hugely depend on scaling
  * Raising to power<1 - sqrt
  * log transformation
  * Scaling - MinMiaxScaler, StandardScaler
  * Rank Transformation - Handles outliers - sets spaces between sorted values to be equal
* When data has outliers - In non-tree models use - rank, llog, sqrt, winsorization


### Feature Generation - Prior Knowledge & EDA






