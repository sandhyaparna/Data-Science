# Parametric and Non-parametric methods:
https://help.xlstat.com/customer/en/portal/articles/2062457-what-statistical-test-should-i-use-?b_id=9283 <br/>
https://www.mayo.edu/research/documents/parametric-and-nonparametric-demystifying-the-terms/doc-20408960 <br/>

The field of statistics exists because it is usually impossible to collect data from all individuals of interest (population). 
Our only solution is to collect data from a subset (sample) of the individuals of interest, but our real desire is to know the “truth” 
about the population. Quantities such as means, standard deviations and proportions are all important values and are called “parameters” 
when we are talking about a population. Since we usually cannot get data from the whole population, we cannot know the values 
of the parameters for that population. We can, however, calculate estimates of these quantities for our sample. When they are calculated 
from sample data, these quantities are called “statistics.” 

### Assumptions
Parametric statistical procedures rely on the shape of distribution of the underlying population. For eg - some parametric tests assume 
that the data should be normally distributed, other tests might assume that the data should be binomial/poission distribution <br/>
Non-parametric procedure doesn't rely on those assumptions.

### Advantages
Parametric: Have more statistical power than non-parametric. It is more able to lead to a rejection of H0 i.e p-value 
is lower compared to that of a corresponding non-parametric p-value <br/>
Non-parametric: More robust than parametric i.e they are valid in broad range of situations


# GLM
GLM are non-linear models where variables are nonlinear, but are still linear in terms of the unknown parameters <br/>
For example, the Cobb-Douglas production function that relates output (Y) to labor (L) and capital (K) can be written as <br/>
            Y = αLβKγ <br/>
Taking logarithms yields <br/>
           ln(Y) = δ + βln(L) + γln(K)      where δ = ln(α)  <br/>
This function is nonlinear in the variables Y, L, and K, but it is linear in the parameters δ, β and γ. <br/> 
Models of this kind can be estimated using the leastsquares technique.  <br/>



