* Metrics to investigate:
  * What hapens if some metric value goes down how does it effect 
  * Why is feature x dropping by y percent?
* Measure the success of a feature or product
  * How do you measure success of FB marketplace or yelp reviews? - The company wants to know if the feature has improved anything and if it was worth the time and effort to build. At this point, the data scientist steps in and thinks of methodology to investigate the data, and determines its success using informed metrics
* Feature change
  * Let's say we want to add/change/improve a new feature to product X. what metrics would you track to make sure it's a good idea?
* Metric Tradeoffs
  * You are product manager for FB reactions. Comments are down by 10% but reactions are up by 15%
* Growth
  * Let's say we want to grow X metric on Y feature. How would we go about doing so?
  
Questions to think about that help to understanding Products better:
* How does LinkedIn make money?
* What are some new features that LinkedIn has released? Why do you think they released them?
* Which features on LinkedIn drew you into the website? Which features drew you into a particular product?
* Why would a person pay money to use a certain service on LinkedIn?

Skills judged during these questions: Think big and always take a step back before you dive in
* Structured Problem Solving
* Decision Making in Ambiguity
* Prioritization Framework
* Impact Sizing
* Stakeholder Management
* Strategic Thinking
* General Business Intuition

Framework for Product Questions:
1. Clarifying the Question </br>
What are the product goals? What’s the background context? We almost always start at an information disadvantage. So, what questions do we need to ask to bridge the gap?
2. Make Assumptions </br>
Make some assumptions about the problem to narrow the scope. State what you’ll explore in your analysis and what you won’t.
3. Analyze User Flows </br>
Examine exactly how the product works. How does a user get to a certain feature? How does a user use a certain feature? What kinds of different users are there?
4. Define Hypothesis </br>
Start hypothesizing situations to explore that would help understand the root cause of the issue.
5. Draw Metrics to Support your Hypothesis </br>
Use metrics as an example to further illustrate how it could prove or disprove your hypothesis. Remember to choose and prioritize which metrics you think are important. Choosing a bad metric that doesn’t represent the problem at hand can oftentimes be a red flag for an interviewer.
6. Tie Your Analysis to the Product Goals </br>
Finally, tie your analysis back to the product goals. Give some sort of summary statement that can prioritize which ones matter and what the next steps are.
7. Remember to talk out loud </br>
Talking out loud helps the interviewer see your thought process and lead you in the right direction if you get lost. </br>
For example, if you don’t know the answer immediately, saying something like “Hm, I’m not exactly sure, but just thinking out loud…”, will help communicate the ambiguity of your initial answers while you figure out the right solution down the line.

### Investigating Metrics
* If X metric is up/down by Y percent, how would you investigate it?
* Use Mutually Exclusive and Collectively Exhaustive framework: https://www.mbacrystalball.com/blog/strategy/mece-framework/
* Clarification: 
  * Is this a one-time event or a progressive movement? 
  * Ask for external factors: include the possibility of a new Facebook competitor launching, or an environmental disaster hitting a large consumer base. Or perhaps today was Saturday and yesterday was Friday– on weekdays, consumers spend more time on social media and thus spend more time sending friend requests. 
  * Ask for internal factors: include bugs or a feature release that caused a user to stop friend requests.
  * Lastly, for clarification, we also have to understand the metric definition. Friend requests sent is not the same as friend requests accepted, and understanding the exact definition for this metric is crucial for an accurate solution.
  * Many times an interviewer will ask you to explain both sides to every question you pose. For example, if it was or was not a bug, what would you then investigate? Lastly, remember to think of high-level common cases of these types of problems first. If you were running Facebook, Google, or any other product, what would be something that would matter to you?
* Gathering Context
  * This step is partly understanding the time frame, as mentioned earlier, but also looking into different correlation effects. We have to understand if this is something that’s happening specifically within this feature, or from external features as well.
  * If Facebook requests are down 10%, is the entire website traffic also down by 10%? 
  * If this is a one-time event, a hypothetical example could be that a new feature places a huge Facebook Marketplace button next to the comparatively smaller friend request button, which might increase marketplace hits at the expense of friend requests.
  * if this is a progressive event, then the question would be more about whether you can analyze the effects of Facebook usage over time. Perhaps this could be due to a behavior change on the platform that is more gradual and due to the growing effect of any AB test variant feature increase or the make-up of user activity and demographics.
* Hyphothesizing and Evaluation
  * One variable we can consider is that if there are fewer new users joining the platform, this may account for fewer friend requests. If we had fewer new users joining the platform, this would imply that fewer friend requests are being made since our hypothesis would be new users would usually be the ones making the most friend requests.
  * segmenting our users by newer users and existing users.
  * Here’s a quick cheat sheet of segmentations we can make.
    * User Attributes
      * Join date
      * Demographic information (Age, Sex, Location, etc…)
      * Acquisition Channel (Organic, Paid, Content, etc..)
      * Geography
      * Platform (iPhone, Desktop, Android, etc..)
    * User Activity Attributes
      * Total Users vs Active Users
      * Daily/Weekly/Monthly Active Users (DAU, WAU, MAU)
      * New vs Existing Users
      * User feature attribution (how they got to the feature)
      * Time spent
      * DAU / MAU or WAU / MAU: daily active users (DAU), weekly active users (WAU), monthly active users (MAU),
    * Marketplace/Feature Attributes (Depends on the problem)
      * Friend requests acceptance rate going down
      * Friend requests taking longer to be accepted
      * Bugs/Broken features
      * New feature cannibalizing the existing feature
  * The key to this analysis is to use your business intuition to understand which segments to investigate and prioritize. It would be impossible to actually analyze everything, so we have to think about the prioritization of important metrics, thereby glossing over the improbable ones and spending more time diving deeper into the more probable ones.
* Evaluation and Validation
  * metrics viz as a graph
  * multiple metrics over time

### Fractional metrics 
* Looking at the metrics of the job board, you notice that the number of applicants per job has been decreasing.
* Clarification
  * Timeframe Is this a one-time event? Is this a progressive event?
  * Metric Definition Is this metric unique or the total? Ex: unique applicants per job versus total applicants per job
  * Decrease/Increase Percentage How big is the decrease or increase in the metric change?
* Gathering Context
  * The numerator got smaller - less total applicants
  * The denominator got bigger - more jobs on the platform
  * For example, let’s say that we propose that the number of job postings has more or less remained the same, but the number of applicants is steadily decreasing. Therefore, we can clarify two things: it’s a progressive decline, and the decline is coming from the numerator getting smaller (the number of applicants decreasing).

### Measuring Success
* How do you measure success for Facebook Events?
* How would you measure success for Yelp reviews?
* How would we measure the success of acquiring new users on Netflix through a free trial?
* How would you measure the success of private stories on Instagram, where only certain chosen friends can see the story?
* What metrics would you track for analyzing the health of Google Docs?










