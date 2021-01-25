










#### Analyze Results
##### Check if everything is okay before testing
* Baseline Metrics: 
  * Before analyzing results, the first step is to do a sanity check — check if your baseline metrics have changed. If the sanity check fails, first go analyze why. We can either perform a retrospective analysis or understand if there’s an error that we can neutralize as an independent variable in our analysis.
  * Check if all the independent variables didn't change during the AB testing or no diff before & after the test to make sure we are evaluating 1 feature while all other features remain constant
* Not significant: If results don't turn out to be significant
  * Break down the data into different segments like platforms or timeframes and see if there is any significance. This may help us understand if there’s a bug in the system, user reaction to the experiment, or if it’s just an insignificant result.
* Significance by Chance: 
  * Cross checking with different methods. For example, we should compare the non-parametric sign test with the parametric hypothesis test. What happens when the hypothesis test and sign test don’t agree?
  * Another observation we should make is that we may be suffering from Simpson’s paradox. The reasons for Simpson’s paradox happening could be that the setup of the experiment is incorrect or the change could affect the new users and experienced users differently.
* Metrics moving in opposite directions: Create a compound metric
  * What if we have two metrics that are moving in opposite directions? To counter this effect, many times we will set one north star metric that is a combination of different metrics. This allows us to balance short term and long term goals in features and track only one metric for testing validity.
##### Tests
* Analyze the results and assess the robustness of the test.
  * Two-tailed t-tests (comparison of means), chi-squared test, and regressions. Think about data transformation & tests
* Present the results of the test to stakeholders!
  * Remember to state any assumptions or caveats involved in the test. If you are presenting to a non-technical audience, remember to keep things explainable!
* Use findings to run another test or change a feature.
  * Remember that A/B testing is used to make decisions. If implementing a ‘winner,’ consider using a holdback in your next experiment, depending on business needs.
##### 
* Don’t try to look for differences for every possible segment.
* Be careful about launching things because they wouldn’t hurt: Sometimes implmenting a feature might have negative impact
* If we have a significant result from the test…: Then comes two questions: 
  * Do you understand the change and do you want to launch the change? What if your change has a positive impact on one slice of users, but no impact or negative impact on other slices of users?
  * Do I have statistically significant and practically significant results to justify the change and the launch? Do I understand what the change did to the user experience? Last but not least, is it worth it to launch?
  




