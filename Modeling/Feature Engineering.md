### Links
https://www.kdnuggets.com/2018/12/feature-building-techniques-tricks-kaggle.html <br/>
https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b <br/>
https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-6-feature-engineering-and-feature-selection-8b94f870706a <br/>

### What makes a good feature?
* Be related to the objective - Have a reasonable hypothesis on why a feature is related to the problem we are solving
* Be know at prediction-time
* Be numeric with meaningful magnitude
* Have enough examples
* Bring human insight to problem 

### Numeric
* Values as it is
* Unique Counts/Freq
* Binning
* Binarize based on a cut-off
* Rounding off - High precision may not be required
* Interaction between variables
* Binning
* Transformation - log, box-cox
* Scaling by Max-Min
* Normalization using Standard Deviation
* Log based feature/Target: use log based features or log based target function


### Categorical
* Encoding
* 

### Binary Variables


### Date and Time
* Day of the week
* Time based Features like "Evening", "Noon", "Night", "Purchases_last_month", "Purchases_last_week" etc.
* Is weekend or not
* Was the date the end of a quarter?, Was the day a holiday?, Were the Olympics/Rare events taking place on said date?
* cash withdrawals can be linked to a pay day; the purchase of a metro card, to the beginning of the month.
* In general, when working with time series data, it is a good idea to have a calendar with public holidays, abnormal weather conditions, and other important events.
* There also exist some more esoteric approaches to such data like projecting the time onto a circle and using the two coordinates.
def make_harmonic_features(value, period=24):
    value * = 2 * np.pi / period 
    return np.cos(value), np.sin(value)

from scipy.spatial import distance
euclidean(make_harmonic_features(23), make_harmonic_features(1)) 
result: 0.517


### Time Series
Generates automatic feature engineering for time series data - https://github.com/blue-yonder/tsfresh <br/>
https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html <br/>


### Web data
* operating system
* Is mobile or not
* Browser
* lag behind the latest version of the browser
* Did user viisted the website previously?
* Count of user views
* number of distinct docs visited by user


### Geographic data - Lat, Lon
* Haversine Distance Between the Two Lat/Lons
* Manhattan Distance Between the two Lat/Lons
* Bearing Between the two Lat/Lons

* Distances (great circle distance and road distance calculated by the routing graph)
* Number of turns with the ratio of left to right turns, number of traffic lights, junctions, and bridges 
* Complexity of the road - graph-calculated distance divided by the GCD
* 
* Proximity of a point to the subway
* Number of stories in the building
* Distance to the nearest store
* Number of ATMs around, etc. 
* Height above sea level, etc. 

### Search based data
* Attribute Features
  * Whether the product contains a certain attribute (brand, size, color, weight, indoor/outdoor, energy star certified …)
  * Whether a certain attribute matches with the search term
* Meta Features
  * Length of each text field
  * Whether the product contains attribute fields
  * Brand (encoded as integers)
  * Product ID
* Matching
  * Whether search term appears in product title / description / attributes
  * Count and ratio of search term’s appearance in product title / description / attributes
  * Whether the i-th word of search term appears in product title / description / attributes
* Text similarities between search term and product title/description/attributes
  * BOW Cosine Similairty
  * TF-IDF Cosine Similarity
  * Jaccard Similarity
  * Edit Distance
  * Word2Vec Distance (I didn’t include this because of its poor performance and slow calculation. Yet it seems that I was using it wrong.)

### Sales Transactional Data
* Most recent transaction related details - when did he buy, what did he buy etc
* Freq
* Monetary
* Average Time Difference between transactions
* Average number of Transactions per month
* Sales
* Number of Disinct Items
* Total number of Transactions
* Percantage of Discounted transactions - total discounted trans/Total Trans
* Maximum Discount Amount
* Total Discount Amount
* Identify first transaction - first. In SAS
* Maximum Sales value
* Minimum Sales Value
* Difference between Time period end date and Last transaction date
* Most Recent Transaction's Sales value
* Average Transactional Value = sales/transactions
* Average items per transaction= Quantity Sold/Transactions
* Volume - Quantity Sold
* Tenure of a Customer (based on their Start Date)
* Age of the firm
* Number of Employees
* Size of the firm (Revenue)
* Location
* Frequent mode of purchase - website/Sales person/ phone
* Participation in Events
* Distance in miles between prospect location and event location






### Automated Feature Engineering - Deep Feature Synthesis
https://docs.featuretools.com/automated_feature_engineering/primitives.html

* Featuretools module in python - Deep feature synthesis
* 

    
    
    
    





