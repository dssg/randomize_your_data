# randomize_your_data
Data leakage is the use of information to train a predictive model that would not be available in production, causing the model to achieve deceptively good performance on historical data. Despite being one of the most recognized and important problems in machine learning, there are few concrete and scalable suggestions for diagnosing leakage, perhaps because there are many possible sources, some of which are subtle. Recommendations typically involve some form of subject matter knowledge, intuition, or interactive data exploration. Here are a few examples:

* "An easy way to know you have data leakage is if you are achieving performance that seems a little too good to be true." ([link](https://machinelearningmastery.com/data-leakage-machine-learning/&sa=D&ust=1533636043512000))
* "Look for surprising feature behavior in the fitted model." ([link](https://www.coursera.org/lecture/python-machine-learning/data-leakage-ois3n&sa=D&ust=1533636043513000))
* Turn "to exploratory data analysis in order to find and eliminate leakage sources." ([link](http://what-when-how.com/Tutorial/topic-83mc5/Doing-Data-Science-335.html&sa=D&ust=1533636043513000))
* "A combination of caution, common sense, and data exploration can help identify leaking predictors so you remove them from your model." ([link](https://www.kaggle.com/dansbecker/data-leakage&sa=D&ust=1533636043514000))

We decided to develop tests to determine whether leakage exists. We think there are at least two basic ways to find leakage: randomly sorting each column and flipping labels. This README discusses these strategies and how they might help detect leakage and points to some of the work we've done addressing this issue. 

## Detection Strategies
We'd like to begin by introducing two simple ideas to help detect leakage: randomizing the raw data and flipping labels.

### Randomize the raw data
If we were to independently and randomly sort each column in the raw data, that variable would retain its univariate properties but the associations between columns would be broken. If we were to model these randomized data, we shouldn't be able to predict anything:   

* The predictions should be no better than random guessing.
* You should not be able to predict an entity's score (relative or absolute) across samples. Another way to put this is that entities ranked high in one sample are no more likely to be highly ranked in another.
* You should not be able to predict feature importances (absolute or relative) across samples.

These are null hypotheses, and deviations from them suggest the existence of leakage. Predictable scores and features may also help identify the source of the leak.

It would be simpler to only randomize the columns that define the label. But label generation can draw on many pieces of information in the raw data in complex ways. Randomizing all the columns in the raw data helps ensure that the data-generating process is completely broken, leading to valid statistical tests.

It is best to randomize as early as possible in the pipeline. Leakage can happen at any step, so early randomization gives you a better chance of detection.

### Test label flipping
If there is leakage from the test set to the train set, your model will anticipate changes in the test set that it should not know about. One way to test for that is to create an exogenous shock in the test set, re-train your models, and observe what happens. Perhaps the simplest way is to flip the test labels, so 1s become 0s and 0s become 1s. Hopefully you'll observe the following:

* The models will be the same---same sizes, same feature importances or coefficients, etc---because the train set should be the same. Models built with deterministic algorithms (e.g. decision trees, logistic regression, random forests built with random seeds) will be exactly the same, while random algorithms (e.g. bagged models) will statistically be the same.
* Train-set performance should remain unchanged but test-set performance should become worse than random. The table below presents a simple example where the model is 80% accurate using the correct labels. Absent leakage, the model is 20% accurate with flipped labels and 80% with. 
* Performance on the flipped test set should be the same as it would be on the non-flipped test set if we had reverse ordered our observations. For example, precision at the bottom k of our flipped test set should be similar to the model's "precision at the top k" using the true test set.  If there's leakage, performance on the flipped test set may be better.

| true test label | flipped test label | predicted label – no leakage | predicted label – leakage|
|:---:|:---:|:---:|:---:|
|0|1|1|0|
|1|0|1|0|
|1|0|1|0|
|1|0|1|0|
|1|0|1|0|
|1|0|0|1|
|0|1|0|1|
|0|1|0|1|
|0|1|0|1|
|0|1|0|1|

All else being equal, flipped labels can provide more statistical power than randomized column orders because the difference between good performance and random performance is smaller than the difference between good performance and bad performance. 

## Sources of Leakage
This section introduces four basic sources of leakage---observations, variables, values, and models---and maps detection strategies to each.

### Observations
Observation leakage occurs when observations from the test set are incorrectly included in the train set. Here are several examples:

#### Using standard cross validation in a non-iid (independent and identically distributed) setting
Most human-related problems have spatial and temporal components, so that a model trained on one spatial or temporal subset performs poorly on another. By randomly mixing observations into the train and test sets, you allow the model to see what it wouldn't in the real world.  

The figure below shows a simple example. The first plot shows data collected from 2012 through 2017. There is a clear shock in 2015. If it were 2014, we might be growing confident in our ability to predict, only to be surprised a year later. Standard cross validation (second plot) would randomly assign these observations to k folds, eliminating the temporal surprise and giving us a false sense of data stability.  

[temporal data](images/image2.png)
[random CV](images.image1.png)

If we want to predict US annual GDP and we have observations from 2015-2017, the three standard sets would be:  
  
* Train 2015-2016, validate 2017
* Train 2015,2017, validate 2016
* Train 2016-2017, validate 2015

Only one of those looks forward, the last one looks backward, but all three are given equal weight.

*Detection strategies*:  
1. Performance should vary with the baserate. Standard cross validation won't show that variation.
2. Flip the test labels.

#### Including entities in the dataset before they are known
If a clinic would like to predict which of its patients will visit in the next year, it cannot include patients whose first visit is at time t+1 in the data at time t. The mere existence of a patient with zero visits at time t perfectly predicts that they will have a visit in the next time period. [Here's](https://medium.com/@colin.fraser/the-treachery-of-leakage-56a2d7c4e931&sa=D&ust=1533636043520000) a similar example for an online store.

*Detection strategies*: We don't have good ideas other than exploring the results with decision trees or other simple models, which might surface strange feature patterns such as "zero past purchases perfectly predict future purchases," and comparing their "knowledge dates" (the first date that the system would have known about them) and their "prediction dates" (in the train and test sets).

#### Your train and test label windows cannot overlap
If we train a model on whether a police officer had an adverse incident between January 1 and December 31, 2015, the train set label covers 2015 (1/1/15 – 12/31/15) and the test set label cannot start until 2016 (1/1/16 or later). Otherwise, an event that affects the test label can also appear in the train label, giving the learner unrealistic insight into the test set.

*Detection strategies*: Flip the labels.

### Variables
There are multiple ways that variables can leak information:  

#### Creating a variable based on the label
The simplest example is to use the label as a variable, but there are more subtle ways for this to happen, such as using the same event to create labels and features.

*Detection strategies*:  

1. Flip the labels.
2. Randomize the data.

#### Using a feature that is not available at the time of prediction
If we're predicting what will happen in 2016, we can't use the 2016 American Community Survey data because it has an almost two-year lag. (We could use 2014 ACS data in 2016, but nothing more recent.) If we're making a prediction for a clinical visit, the model can't use information from the labs taken during the visit because the results aren't available until later.

*Detection strategies*: We don't have good ideas for detection, other than subject-matter expertise. Good database design can help. Ideally, every row will have timestamps to indicate when the information was known or gathered. Each column's associated timestamps should precede the row date (the date of the prediction that the row represents). A recent article suggested using "[legitimacy tags](https://www.researchgate.net/publication/221653692_Leakage_in_Data_Mining_Formulation_Detection_and_Avoidance)."

#### Selecting features using the entire dataset and then conducting cross validation
That allows your model to know something about the test sets that it shouldn't.

*Detection strategies*: randomize your data.

### Cell values
Cell values may leak information if they are incorrect:

In a previous job, one of the authors was tasked with predicting quarterly sales for an industry. He received corporate sales data and went to work, not knowing that part of the data had accidentally been copied and pasted four quarters ahead, leading to unusually good results.

*Detection strategies*: We don't have any good ideas for this type of leakage. It may require knowing the true numbers or having an intuition for overly optimistic results. 

Leakage can occur through imputation – basically replacing missing values with informed guesses – and outlier modification if done on the entire dataset.

*Detection strategies*: Flip the labels in the raw test data and see if the imputed feature values change on the train set. 

### Models
Models implicitly impose assumptions about the relationships between observations, variables, and cell values. They're another vehicle for cheating. For example, experts often know something about patterns in the data. Allowing them to choose model types and priors for data they know is a form of cheating. Let's say we're building a model to predict nominal GDP in the following year. Having economics backgrounds, the authors know that GDP declined by $300 billion in 2009, so we could simply add a dummy variable for that year and force it to have a negative value. It might look like this:

*GDP<sub>t</sub> = a + GDP<sub>t-1</sub> - $300 billion * 1[year is 2009] + e<sub>t</sub>*  
  
This model cheats because we would not have known GDP was going to decline by $300 billion if you asked us in, say, 2006 or 2007.

*Detection strategies*:

1. Randomize data.
2. Flip labels.
3.	The model performs as well, or even better, on test-set "surprises": hard-to-predict cases such as a decline in GDP.
4.	Compare the performance of "simple" models (which are easier to influence with expert information) and "complex" models (which are more difficult to influence with expert information). Complex models typically beat simple models for real-world data.

## Hypothesis Tests
You could use standard statistical testing, where you require a level of evidence that leakage exists. But for some problems, you might want to be more conservative and require a level of evidence that leakage does not. Carlisle Rainey's "two one-sided tests" ([TOST](http://www.carlislerainey.com/papers/nme.pdf)) provide a solution. Find examples in the code.

## References:  
- [https://www.kaggle.com/wiki/Leakage](https://www.kaggle.com/wiki/Leakage)
- [https://www.kaggle.com/dansbecker/data-leakage](https://www.kaggle.com/dansbecker/data-leakage)
- [https://www.coursera.org/learn/python-machine-learning/lecture/ois3n/data-leakage](https://www.coursera.org/learn/python-machine-learning/lecture/ois3n/data-leakage)
- [https://medium.com/@colin.fraser/the-treachery-of-leakage-56a2d7c4e931](https://medium.com/@colin.fraser/the-treachery-of-leakage-56a2d7c4e931)
- [http://dstillery.com/wp-content/uploads/2014/05/Leakage-in-Data-Mining-Formulation-Detection-and-Avoidance.pdf](http://dstillery.com/wp-content/uploads/2014/05/Leakage-in-Data-Mining-Formulation-Detection-and-Avoidance.pdf)

## Repo Contents
- randomization\_strategy: Exploration of the randomization strategy for leakage detection using dummy data and models.
- dirty\_duck: Application of randomization strategy toward the [Dirty Duck Tutorial](https://github.com/dssg/dirtyduck), using City of Chicago restaurant-inspection data.
- la\_prosecutor: A series of tests for leakage, demonstrated on the Los Angeles Chronic Offenders project.
