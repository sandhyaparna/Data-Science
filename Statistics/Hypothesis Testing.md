### Differnt Tests
https://help.xlstat.com/customer/en/portal/articles/2062457-what-statistical-test-should-i-use-?b_id=9283
Statistics in a nutshell - Table 5-27,Summary of tests covered in this chapter 

### Hypothesis testing 
It is the use of statistics to determine the probability that a given hypothesis is true. The usual process of hypothesis testing consists of four steps:
1. Formulate the null hypothesis H_0 (commonly, that the observations are the result of pure chance) and the alternative hypothesis H_a (commonly, that the observations show a real effect combined with a component of chance variation; Sample observations are influenced by some non-random cause).
2. Identify a test statistic that can be used to assess the truth of the null hypothesis.
3. Compute the P-value, which is the probability that a test statistic at least as significant as the one observed would be obtained assuming that the null hypothesis were true. The smaller the P-value, the stronger the evidence against the null hypothesis.
4. Compare the p-value to an acceptable significance value  alpha (sometimes called an alpha value). If p<=alpha, that the observed effect is statistically significant, the null hypothesis is ruled out, and the alternative hypothesis is valid.

##### Level of significance
Refers to the degree of significance in which we accept or reject the null-hypothesis.  100% accuracy is not possible for accepting or rejecting a hypothesis, so we therefore select a level of significance that is usually 5%.

### Decision Errors
##### Type I error
A Type I error occurs when the researcher rejects a null hypothesis when it is true. The probability of committing a Type I error is called the significance level. This probability is also called alpha, and is often denoted by α.
##### Type II error
A Type II error occurs when the researcher fails to reject a null hypothesis that is false. The probability of committing a Type II error is called Beta, and is often denoted by β. The probability of not committing a Type II error is called the Power of the test.

### Decision Rules
In hypothesis testing, we either reject null hypothesis or fail to reject null hypothesis. <br/>
The analysis plan includes decision rules for rejecting the null hypothesis. In practice, statisticians describe these decision rules in two ways - with reference to a P-value or with reference to a region of acceptance. <br/>
These approaches are equivalent <br/>
##### P-value
The strength of evidence in support of a null hypothesis is measured by the P-value. Suppose the test statistic is equal to S. The P-value is the probability of observing a test statistic as extreme as S, assuming the null hypotheis is true. If the P-value is less than the significance level, we reject the null hypothesis.
##### Region of acceptance
The region of acceptance is a range of values. If the test statistic falls within the region of acceptance, the null hypothesis is not rejected. The region of acceptance is defined so that the chance of making a Type I error is equal to the significance level. <br/>
The set of values outside the region of acceptance is called the region of rejection. If the test statistic falls within the region of rejection, the null hypothesis is rejected. In such cases, we say that the hypothesis has been rejected at the α level of significance. <br/>

### One-Tailed and Two-Tailed Tests
##### One-Tailed
A test of a statistical hypothesis, where the region of rejection is on only one side of the sampling distribution, is called a one-tailed test. For example, suppose the null hypothesis states that the mean is less than or equal to 10. The alternative hypothesis would be that the mean is greater than 10. The region of rejection would consist of a range of numbers located on the right side of sampling distribution; that is, a set of numbers greater than 10.
* Left-tailed test: Population parameter is less than a certain value
* Right-tailed test: Population parameter is greater than a certain value

##### Two-Tailed Tests
Population parameter is not equal to a certain value <br/>
A test of a statistical hypothesis, where the region of rejection is on both sides of the sampling distribution, is called a two-tailed test. For example, suppose the null hypothesis states that the mean is equal to 10. The alternative hypothesis would be that the mean is less than 10 or greater than 10. The region of rejection would consist of a range of numbers located on both sides of sampling distribution; that is, the region of rejection would consist partly of numbers that were less than 10 and partly of numbers that were greater than 10. <br/>

Welch’s t-test is used in case homogenity assumption is not met.

### Ho: Means are identical 
Quantitative variable is dependent and Categorical vars are independent
* One Sample T-Test: 1 Quantitative variable
* T-Test on two independent samples: 1 Categorical var with 2 categories, 1 Quantitative variable
* T-test on two paired samples: 1 Categorical var with 2 categories i.e Before and After, 1 Quantitative variable
* One-way ANOVA: 1 Categorical var with 2+ categories, 1 Quantitative variable
* Factorial ANOVA (2-way, 3-way etc.): 2 or more Categorical vars & interaction between the 2, with 2 or more categories, 1 Quantitative variable
* Multiple ANOVAs: 1 Categorical var with 2 or more categories, 2 or more Quantitative variables
* MANOVA: 2 or more Categorical vars with 2 or more categories, 2 or more Quantitative variables

### Ho: Variance
* Fisher's test: Comparision between 2 variances
* Levene's test: Comparision between 2+ variances 

### Ho: Proportions (Categorical Var)
Exact Fisher test is performed if Total N is <=90  (or) expected freq less than 5 <br/>
Chi-sq test is performed only when atleast 80% of the cells have an expected frequency of 5 or greater <br/>
* chi-square goodness of fit: Tests if a proportion is equal to a theoritical proportion
* chi-square: 1 Categorical var with 2+ categories
* Multinomial Goodness-Of-Fit test: 
* Chi-square on contingency table: 2 Categorical vars with 2 or more categories (If variables are independent or not)
##### Correlation stats for Categorical data
* Phi measures the degree of association between two binary variables (two categorical variables, each of which can have only one of two values). Phi is calculated for 2×2 tables. Phi = (chi-sq/n)^0.5
* Cramer’s V is analogous to phi for tables larger than 2×2. Cramer’s V = (chi-sq/n* min(r-1,c-1))^0.5
* Point-Biserial Correlation Coefficient is a measure of association between a dichotomous variable and a continuous variable

### Ho: Correlation
Correlation values ranges from -1 to +1 <br/>
Checks if 2 samples are related (linear relationship)
* Pearson Correlation Test - Parametric
* Spearman's correlation Test - NonParametric
* Goodman and Kruskal’s gamma - measure of association for ordinal variables that is based on the number of concordant and
discordant pairs between two variables. It is sometimes called a measure of monotonicity because it tells you how often the variables have values in the order expected
* Kendall’s Tau Rank-Order Correlation
#### Assumptions of Pearson Correlation Test
* Observations in each sample are independent and identically distributed (iid)
* Observations in each sample are normally distributed
* Observations in each sample have the same variance
#### Assumptions of Spearman's correlation Test - Used for Ordinal data
* Observations in each sample are independent and identically distributed (iid)
* Observations in each sample can be ranked

### Ho: Normality
* Shapiro-Wilk: Data sample has Gaussian distribution or not
* D’Agostino’s K^2: Data sample has Gaussian distribution or not - It is based on transformations of the sample kurtosis and skewness
* Anderson-Darling: Data sample has Gaussian distribution or not - Most powerful test for Normality. 
##### Assumptions for all the Normality Tests
Observations in each sample are independent and identically distributed (iid)

-------

### One Sample T-Test
The one sample t-test is a statistical procedure used to determine whether a sample of observations could have been generated by a process with a specific mean
The alternative hypothesis assumes that some difference exists between the true mean (μ) and the comparison value (m0), whereas the null hypothesis assumes that no difference exists.
##### Assumptions
* The dependent variable must be continuous (interval/ratio).
* The observations are independent of one another.
* The dependent variable should be approximately normally distributed - Histogram
* The dependent variable should not contain any outliers - BoxPlot

### T-Test on two independent samples
One independent, categorical variable that has two levels/groups & One continuous dependent variable. <br/>
The independent t-test, also called the two sample t-test, independent-samples t-test or student's t-test, is an inferential statistical test that determines whether there is a statistically significant difference between the means in two unrelated groups. <br/>
##### Assumptions
* The dependent var is approximately normally distributed within each group
  * Mann-Whitney U Test doesn't require Normality assumption
* Homogeneity of variances - The variances of the two groups you are measuring are equal in the population - Levene’s test for homogeneity
  * Adjustment to degrees of freedom using the Welch-Satterthwaite method, Welch t Test statistic
* Independence of observations - No relationship between observations in each group or between the groups themselves
* No outliers - BoxPlot

### ANOVA
Analysis of variance (ANOVA) is a statistical procedure used to compare the mean values on some variable between two or more independent groups. Uses F-statistic.
##### Assumptions
* The dependent variable must be continuous (interval/ratio).
* The observations are independent of one another.
* Each group sample is drawn from a normally distributed population
* All populations have a common variance
* Size of eachgroup is >20

### Pearson Correlation 
Measures the strength & direction of LINEAR relationship between 2 variables
##### Assumptions
* Observations in each sample are independent and identically distributed (iid)
* Observations in each sample are normally distributed
* Observations in each sample have the same variance

### Spearman's Rank-Order Correlation
It is the nonparametric version of the Pearson correlation <br/>
Measures strength & direction of MONOTONIC realtionship between 2 ranked variables <br/>
A monotonic (need not be linear) relationship is a relationship that does one of the following: (1) as the value of one variable increases, so does the value of the other variable; or (2) as the value of one variable increases, the other variable value decreases <br/>
Spearman’s rho: usually have larger values than Kendall’s Tau.  Calculations based on deviations.  Much more sensitive to error and discrepancies in data
##### Assumptions
* Observations in each sample are independent and identically distributed (iid)
* Observations in each sample can be ranked(oridinal), interval or ratio

### Kendall’s Tau Rank-Order Correlation
Measures strength & direction of MONOTONIC realtionship between 2 ranked variables <br/>
Kendall’s Tau: usually smaller values than Spearman’s rho correlation. Calculations based on concordant and discordant pairs. Insensitive to error. P values are more accurate with smaller sample sizes <br/>
Concordant pairs: If both members of one observation are larger than their respective members of the other observations <br/>
Discordant pairs: If the two numbers in one observation differ in opposite directions br/>
##### Assumptions
* Observations in each sample are independent and identically distributed (iid)
* Observations in each sample can be ranked(oridinal), interval or ratio

### Chi-Squared Test
Tests whether two categorical variables are related or independent
##### Assumptions
* Observations used in the calculation of the contingency table are independent
* 25 or more examples in each cell of the contingency table
Expected freq = (Row Total * Column Total)/Overall Total
Chi-square = Σ((Xobs-Xexpected)/Xexpected)




Chi sq goodness of fit - if a sample proportion is consistent with hypothesized distribution
chi sq test - if 2 categorical vars are idependent or not?

Diff between chi-sq and Fischer exact test
