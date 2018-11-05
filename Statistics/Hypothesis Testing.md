https://help.xlstat.com/customer/en/portal/articles/2062457-what-statistical-test-should-i-use-?b_id=9283

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

### Ho: Means are identical 
* One Sample T-Test: 1 Quantitative variable
* T-Test on two independent samples: 1 Categorical var with 2 categories, 1 Quantitative variable
* T-test on two paired samples: 1 Categorical var with 2 categories i.e Before and After, 1 Quantitative variable
* One-way ANOVA: 1 Categorical var with 2+ categories, 1 Quantitative variable
* Factorial ANOVA: 2 or more Categorical vars with 2 or more categories, 1 Quantitative variable
* Multiple ANOVAs: 1 Categorical var with 2 or more categories, 2 or more Quantitative variables
* MANOVA: 2 or more Categorical vars with 2 or more categories, 2 or more Quantitative variables

### Ho: Equal Variance
* Fisher's test: Comparision between 2 variances
* Levene's test: Comparision between 2+ variances 

### Ho: Proportions (Categorical Var)
Exact Fisher test is performed if Total N is <=90  (or) expected freq less than 5 <br/>
Chi-sq test is performed only when atleast 80% of the cells have an expected frequency of 5 or greater <br/>
* chi-square: Tests if a proportion is equal to a theoritical proportion
* chi-square: 1 Categorical var with 2+ categories
* Multinomial Goodness-Of-Fit test: 
* Chi-square on contingency table: 2 Categorical var with 2 or more categories (If variables are independent or not)

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
* The dependent is approximately normally distributed within each group
  * Mann-Whitney U Test doesn't require Normality assumption
* Homogeneity of variances - The variances of the two groups you are measuring are equal in the population - Levene’s test for homogeneity
  * Adjustment to degrees of freedom using the Welch-Satterthwaite method, Welch t Test statistic
* Independence of observations - No relationship between observations in each group or between the groups themselves
* No outliers - BoxPlot





Chi sq goodness of fit - if a sample proportion is consistent with hypothesized distribution
chi sq test - if 2 categorical vars are idependent or not?

Diff between chi-sq and Fischer exact test
