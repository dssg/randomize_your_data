# randomize_your_data
Randomize the order of each column of raw data to help check for leakage. If model still has predictive power after all input columns have been randomized, this is a sign that leakage exists somewhere in the pipeline.  

Type of leakage that may be detected with this method: Leaky validation strategies, i.e. information is leaked somewhere in the machine learning pipeline. For example:
- Leakage of test data into training data during preprocessing / cleaning
- Pre-selecting features based on entire dataset
- Including the correct label in the feature set
- Incorrect handling of temporal data, leaking information from the future into the past -- only when time/period info is available as indicator column(s)

Type of leakage that would not be detected: Leaky predictors, i.e.predictors contain information that should not be available at decision time. For example: 
- Leakage from columns that serve as proxies for the outcome
- Using external data sources not available to model's natural environment 

Data leakages references:  
- https://www.kaggle.com/wiki/Leakage
- https://www.kaggle.com/dansbecker/data-leakage  
- https://www.coursera.org/learn/python-machine-learning/lecture/ois3n/data-leakage
- https://medium.com/@colin.fraser/the-treachery-of-leakage-56a2d7c4e931
- http://dstillery.com/wp-content/uploads/2014/05/Leakage-in-Data-Mining-Formulation-Detection-and-Avoidance.pdf

Repo contents:
- randomization_strategy: Exploration of the randomization strategy for leakage detection using dummy data and models.
- dirty_duck: Application of randomization strategy towards the Dirty Duck Tutorials[https://github.com/dssg/dirtyduck], using City of Chicago inspections prioritization data.
- la_prosecutor: A series of tests for leakage, demonstrated on the Los Angeles Chronic Offenders project.
