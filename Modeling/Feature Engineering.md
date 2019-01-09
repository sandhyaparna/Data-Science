### Links
https://www.kdnuggets.com/2018/12/feature-building-techniques-tricks-kaggle.html
https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-6-feature-engineering-and-feature-selection-8b94f870706a


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
Generates automatic feature engineering for time series data - https://github.com/blue-yonder/tsfresh
https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html


### Web data
* operating system
* Is mobile or not
* Browser
* lag behind the latest version of the browser
* 


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



    
### Automated Feature Engineering - Deep Feature Synthesis
https://docs.featuretools.com/automated_feature_engineering/primitives.html

* Featuretools module in python - Deep feature synthesis
* 

    
    
    
    





