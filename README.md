# Enron Person of Interest Identifier
Date Reviewed: 12/11/2017

## Summary

The goal of this project is to build a person of interest identifier based on financial and email data made public as a result of the Enron scandal. A person of interest is an individual who was indicted, reached a settlement of plea deal with the government, or testified in exchange for prosecution immunity. Machine learning is useful in trying to accomplish this goal because it can utilize the significant amount of typically confidential information of top Enron's excecutives that entered into the public record, including tens of thousands of emails and detailed financial data for top executives. A machine learning model will categorize the data, find trends in them, and apply these information to new datasets.

## Installation

To clone this repository into a local repository, do:
```
$ git clone https://github.com/Acheh/EnronPOIIdentifier.git
$ cd EnronPOIIdentifier
```

To see the HTML report file, open it in your browser.
To see the Jupyter Notebook ipynb file used to generate the report, open it in Jupyter Notebook.
To get the PoI identifier, run poi_id.py. Make sure you have the following packages installed:
- pickle
- pandas
- numpy
- matplotlit.pyplot
- time
- featureFormat, targetFeatureSplit from feature_format
- Pipeline, FeatureUnion from sklearn.pipeline
- MinMaxScalar, Normalizer from sklearn.preprocessing
- SelectKBest from sklearn.feature_selection
- PCA from sklearn.decomposition
- GridSearchCV from sklearn.model_selection
- StratifiedShuffleSplit from sklearn.cross_validation
- GaussianNB from sklearn.naive_bayes
- DecisionTreeClassifier from sklearn.tree
- RandomForestClassifier, AdaBoostClassifier from sklearn.ensemble

## What's Done in This Project
- data cleaning, outliers removal
- feature engineering
- manual and automated feature selection using SelectKBest and/or PCA
- parameter tuning using Pipeline and GridSearchCV
- cross validation using StratifiedShuffleSplit
- model training using GaussianNB, DecisionTreeClassifier, RandomForestClassifier, and AdaBoostClassifier

## Result
### Feature Engineering:
* bonus_fraction : a fraction of bonus to the salary, which might be more useful than simply bonus, because a higher bonus-fraction might indicate fraud or bribery.
* from_poi_fraction : a fraction of the number of e-mails received from poi to the number of total e-mails received, which is more telling of how often a person communicate with poi than non-poi in general.
* to_poi_fraction : a fraction of the number of e-mails sent to poi to the number of total e-mails sent, which is also more telling of how often a person communicate with poi than non-poi in general.

### Feature Selection using SelectKBest(k=6)
| Feature            | SelectKBest Score | SelectKBest P-value | DecisionTreeClassifier Feature Importance |
|--------------------|:-----------------:|:-------------------:|:-----------------------------------------:|
| other              |03.15              |0.08                 |0.39                                       |
| expenses           |03.66              |0.06                 |0.27                                       |
| to_poi_fraction    |12.41              |0.00                 |0.11                                       |
| from_poi_fraction  |01.89              |0.17                 |0.08                                       |
| deferred_income    |09.01              |0.00                 |0.08                                       |
| bonus_fraction     |07.73              |0.01                 |0.08                                       |

### Final Classifier:
**DecisionTreeClassifier**(min_samples_leaf=4, max_depth=5, class_weight=balanced)
- Accuracy: 0.81085
- Precision: 0.42028
- Recall: 0.60500
- F1: 0.49600
- F2: 0.55612
- Total predictions: 13000
- True positives: 1210
- False positives: 1669
- False negatives:  790
- True negatives: 9331
