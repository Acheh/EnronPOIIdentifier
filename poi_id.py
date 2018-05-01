# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:20:46 2017

@author: aneuk
"""

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from sklearn.pipeline import Pipeline, FeatureUnion    
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

###############################################################################
# VARIABLES & PARAMETERS                                                      #
###############################################################################

# Set timer to time loading time
t1 = time.time()

# Load the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# List of available features
available_features = data_dict[data_dict.keys()[0]].keys()
available_features.remove('poi')
available_features.remove('email_address')
available_features.remove('total_payments')
available_features.remove('total_stock_value')

# Number of available features
n_available_features = len(available_features)

# Number of original data points
n_data_points = len(data_dict)

# Number of POI and Non-POI in dataset
n_poi = sum(data_dict[p]['poi'] for p in data_dict)
n_non_poi = n_data_points - n_poi

# Portion of missing values in each features
portion_nan_in_features = {}
for feature in available_features:
    nan = [1 if data_dict[p][feature] == 'NaN' else 0 for p in data_dict]
    portion_nan_in_features[feature] = round(sum(nan)*1./n_data_points,2)

# Count of missing values in each data points    
count_nan_in_person = {}
for p in data_dict:
    nan = [1 if data_dict[p][key] == 'NaN' else 0 for key in available_features]
    count_nan_in_person[p] = sum(nan)
    
# The outliers that need to be removed from the dataset. 
outliers_list = [
        'TOTAL',
        'THE TRAVEL AGENCY IN THE PARK',
        ]

# Initial features selected (manual feature selection based on missing values
# and feature definition). Note that these features will be used for tuning 
# different classifiers.
initial_features = ['poi']
# The original features selected:
original_features_selected = [
         'deferral_payments', 
         'deferred_income', 
         'exercised_stock_options', 
         'expenses', 
         'long_term_incentive',
         'other', 
         'restricted_stock_deferred',
         'salary'
        ]
# The newly engineered features selected:
new_features = [
        'bonus_fraction',
        'from_poi_fraction',
        'to_poi_fraction'
        ]
initial_features.extend(original_features_selected)
initial_features.extend(new_features)

# Final features selected (manual and automated feature selection). Note that 
# these features are selected based on the result of manual and automated
# feature selection after tuning different classifiers and will be used for
# tuning the final identifier
features_list = [
        'poi',
        'deferred_income',         
        'expenses',
        'other',
        'bonus_fraction',
        'from_poi_fraction',
        'to_poi_fraction'
        ]

# Pipeline Parameters to tune each classifiers. The pipeline used in this 
# program has 3 steps: feature_scaling, feature_selection, and clf.
# You can change/tune these params to different values or range to get a 
# better/worse result. 
# Those listed below produced best scores for me.

# GaussianNB Parameters
gnb_scaler_params = None
gnb_selection_params = {
        'feature_selection__kbest__k': [8],
        'feature_selection__pca__n_components': [2]
        }
gnb_clf_params = None

# DecisionTreeClassifier Parameters
dtree_scaler_params = None
dtree_selection_params = {
        'feature_selection__k': [9]
        }
dtree_clf_params = {
        'clf__min_samples_leaf': [4],
        'clf__max_depth': [5],
        'clf__class_weight': ['balanced']
        }

# RandomForestClassifier Parameters
rforest_scaler_params = None
rforest_selection_params = {
        'feature_selection__k': [9]
        }
rforest_clf_params = {
        'clf__n_estimators': [50],
        'clf__criterion': ['entropy'],
        'clf__min_samples_leaf': [1]
        }

# AdaBoostClassifier Parameters
aboost_scaler_params = None
aboost_selection_params = {
        'feature_selection__k': [9]
        }
aboost_clf_params = {
        'clf__base_estimator': [
                DecisionTreeClassifier(
                        max_depth = 1,
                        min_samples_leaf = 3,
                        class_weight='balanced'
                        )
                ],
        'clf__n_estimators': [100],
        'clf__learning_rate': [0.1]
        }

###############################################################################
# FUNCTIONS FOR DATA EXPLORATION AND DATA CLEANING                            #
###############################################################################
def print_data_features(features_list):
    '''
    Print feature list
    Args:
        features_list : list of features
    '''
    for f in features_list:
        if f != 'poi':
            print "- {}".format(f)

def plot_nan_features():
    '''
    Plot a bar graph of the portion of missing values in each features in
    the original dataset.
    '''
    nan_features_series = pd.Series(
            portion_nan_in_features).sort_values(ascending=True)
    nan_features_series.plot.bar(
            title='Portion of Missing Values in Features',
            y='(%)')
    plt.show()

def print_nan_datapoints(min_nan = 15):
    '''
    Print each data points that has missing values more than min_nan
    Args:
        min_nan: minimum nan count in each data points. Default = 15.
    '''
    print "{:30} Missing Values out of 17 features.".format("Name")
    for p in count_nan_in_person:
        if count_nan_in_person[p] > min_nan:
            print "{:30}: {}".format(p, count_nan_in_person[p])

def print_outliers(feature_name, lower=False):
    '''
    Print data outliers for a certain feature.
    Args:
        feature_name: the name of the feature
        lower: if True, include the lower bound outliers. Default: False.
    '''
    tupple = [(p, data_dict[p][feature_name]) for p in data_dict]
    
    series = pd.Series([tupple[i][1] for i in range(len(tupple))])
    series = series[series!='NaN']
    iqr = series.quantile(.75) - series.quantile(.25)
    upper_bound = series.quantile(.75) + 1.5 * iqr
    lower_bound = series.quantile(.25) - 1.5 * iqr
    
    for item in tupple:
        if item[1] != 'NaN':
            if item[1] > upper_bound:
                print '{:30}: {:10}'.format(item[0], item[1])
            if lower and item[1] < lower_bound:
                print '{:30}: {:10}'.format(item[0], item[1])

def plot_poi_scatter(data,
                     x_feature = 'salary',
                     y_feature = 'bonus',
                     x_label = "x", 
                     y_label = "y", 
                     save_file=False):
    """
    Create a scatter plot from data.
    Args:
        data: the dataset
        x_feature: feature name for x-axis. Default: 'salary'.
        y_feature: feature name for y-axis. Default: 'bonus'.
        x_label: label for the x-axis. Default: "x".
        y_label: label for the y-axis. Default: "y".
        save_file: default False. If true, save the scatter plot in a png file 
                   to a local directory
    """
    point = np.array([[data[p][x_feature],
                       data[p][y_feature],
                       data[p]['poi']] for p in data])
    
    for x, y, poi in point:
        if poi == 'True':
            plt.scatter(x, y, color='r')
        else:
            plt.scatter(x, y, color='b')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
    if save_file:
        fig_name = x_label + "_" + y_label + "_scatter.png"
        plt.savefig(fig_name, transparent = True)
    plt.show()

def calculate_fraction(devidend, divisor):
    """
    Calculate the fraction of between devidend and divisor. This is used for
    creating new features.
    Args:
        devidend: the devidend of the fraction.
        divisor: the divisor of the fraction.
    Returns:
        fraction or NaN
    """
    if devidend != 'NaN' and divisor != 'NaN':
        if divisor != 0:
            return round(devidend * 1. / divisor, 2)
        
    return 'NaN'

def prepare_data(dataset):
    '''
    Clean the dataset from outliers and create additional features
    Args:
        dataset: the dataset to be cleaned
    Returns:
        cleaned dataset
    '''
    # 1. Remove Outliers
    for outlier in outliers_list:
        del dataset[outlier]
            
    # 2 Creating additional features
    for person in dataset:
        # New bonus_fraction feature
        bonus = dataset[person]['bonus']
        salary = dataset[person]['salary']
        
        dataset[person]['bonus_fraction'] = \
        calculate_fraction(bonus, salary)
        
        # New from_poi_fraction feature
        to_messages = dataset[person]['to_messages']
        from_poi = dataset[person]['from_poi_to_this_person']
        
        dataset[person]['from_poi_fraction'] = \
        calculate_fraction(from_poi, to_messages)
            
        # New to_poi_fraction feature
        from_messages = dataset[person]['from_messages']
        to_poi = dataset[person]['from_this_person_to_poi']
                
        dataset[person]['to_poi_fraction'] = \
        calculate_fraction(to_poi, from_messages)
        
    return dataset

def print_summary():
    '''
    Print the summary of the original and new dataset.
    '''
    print 'Available Features:'
    print_data_features(available_features)
    print ''
    
    print 'New Features:'
    print_data_features(new_features)
    print ''
    
    print 'There are originally {} data points in the dataset.'.format(
            n_data_points)
    print 'The following outliers are removed from the dataset:'
    for outlier in outliers_list:
        print '  - {}'.format(outlier)
    print 'Leaving {} data points in the final dataset.'.format(
            n_data_points-len(outliers_list))
    print ''
    print 'Below are the scatter plot for salary vs. bonus.'
    print 'Before outliers removal:'
    plot_poi_scatter(data_dict, x_label = "Salary", y_label = "Bonus")
    print ''
    print 'After outliers removal:'
    plot_poi_scatter(my_dataset, x_label = "Salary", y_label = "Bonus")
    print ''
    
    print 'There are many NaN missing values in features as shown in the graph:'
    plot_nan_features()
    print ''
    print 'Based on missing values and feature definitions,'
    print 'the following features are initally selected:'
    print_data_features(initial_features)
    print ''

###############################################################################
# CLASS AND FUNCTION FOR TUNING DIFFERENT CLASSIFIERS                         #
###############################################################################
class ClassifierTuner():
    '''
    This class is used to create a classier tuner. It is used to build a 
    Pipeline, run GridSearchCV, and test the classifier.
    '''
    # parameters for feature scaling
    scaler_params = None
    # parameters for feature selection
    selection_params = None
    # parameter for classifier
    clf_params = None
    # the Pipeline object
    pipe = None
    # the GridSearchCV object
    grid_search = None
    
    def __init__(self, scaler=None, selection=None, clf=None):
        '''
        Initialize a ClassifierTuner object
        Args:
            scaler : if None, no feature scaling is used. 
                     Supported feature scaling:
                         - MinMaxScaler
                         - Normalizer
                     Default: None.
            selection : if None, Feature Union is used for feature selection.
                        Supported feature selection:
                            - SelectKBest
                            - PCA
                            - Feature Union using SelectKBest and PCA
                        Default: None.
            clf: if None, GaussianNB is used for classifier.
                 Supported classifier:
                     - DecisionTreeClassifier
                     - AdaBoostClassifier
                     - RandomForestClassifier
                     - GaussianNB
                Default: None.
        '''
        # Feature Scaling
        if scaler is not None and (
                not isinstance(scaler, MinMaxScaler) or
                not isinstance(scaler, Normalizer)):
            self.scaler = None
        else:
            self.scaler = scaler
        
        # Feature Selection
        if selection is None or (
                not isinstance(selection, PCA) and 
                not isinstance(selection, SelectKBest) and 
                not isinstance(selection, FeatureUnion)):
            self.selection = \
            FeatureUnion([
                    ('pca',PCA(n_components=1)),
                    ('kbest', SelectKBest(k=1))])
            self.selection_params = {
                    'feature_selection__kbest__k': [1],
                    'feature_selection__pca__n_components': [1]}
        else:
            self.selection = selection
        
        # Classifier
        if clf is None or (
                not isinstance(clf, GaussianNB) and 
                not isinstance(clf, RandomForestClassifier) and 
                not isinstance(clf, AdaBoostClassifier) and 
                not isinstance(clf, DecisionTreeClassifier)):
            self.clf = GaussianNB()
        else:
            self.clf = clf
            
    def build_pipe(self):
        '''
        Create a Pipeline based on ClassifierTuner's feature scaling, feature
        selection, and classifier.
        '''
        estimator = []
        
        if self.scaler is not None:
            estimator += [('feature_scaling', self.scaler)]
        estimator += [('feature_selection', self.selection)]
        estimator += [('clf', self.clf)]
        
        try:
            self.pipe = Pipeline(estimator)
        except:
            print "Cannot build pipe"
        
    def run_GridSearchCV(self, data, features_list, folds=20):
        '''
        Create a GridSearchCV based on ClassifierTuner's Pipeline, feature
        scaling parameters, feature selection parameters, and classifier
        parameters.
        Args:
            data: the dataset
            features_list: the features
            folds: number of folds for cross validation. Default: 20.
        '''
        
        if self.pipe is None:
            self.build_pipe()
            
        if self.pipe:
            data = featureFormat(data, features_list, sort_keys = True)
            labels, features = targetFeatureSplit(data)
            
            param_grid = {}
            if self.scaler and self.scaler_params:
                param_grid.update(self.scaler_params)
            if self.selection_params:
                param_grid.update(self.selection_params)
            if self.clf_params:
                param_grid.update(self.clf_params)
            
            # Determine Cross Validation method.
            cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
        
            # Tune the classifier. Uncomment the scoring param to run the 
            # GridSearchCV based on F1 score, but it will produce warning for
            # ill-defined F-score due too one label prediction. 
            self.grid_search = GridSearchCV(self.pipe,
                                            param_grid=param_grid,
                                            #scoring='f1',
                                            cv=cv)
            
            if self.grid_search:
                # Train the classifier. 
                self.grid_search.fit(features, labels)
                
    def test_best_estimator(self, data, features_list):
        '''
        Test the best estimator from GridSearchCV.
        Args:
            data: the dataset
            features_list: the features
        '''
        if self.grid_search is None:
            print "Please run GridSearchCV before testing"
        else:
            test_classifier(self.grid_search.best_estimator_,
                            data,
                            features_list)
            
def tune_classifiers(data, features):
    '''
    Create different classifier tuners and tune these classifiers
    Args:
        data : dataset
        features : selected features
    Returns:
        GaussianNB classifier tuner
        DecisionTreClassifier classifier tuner
        RandomForestClassifier classifier tuner
        AdaBoostClassidier classifier tuner
    '''
    # GaussianNB
    gnb_tuner = ClassifierTuner()
    gnb_tuner.scaler_params = gnb_scaler_params
    gnb_tuner.selection_params = gnb_selection_params
    gnb_tuner.clf_params = gnb_clf_params
    
    gnb_tuner.build_pipe()
    t0 = time.time()
    gnb_tuner.run_GridSearchCV(data, features, 20)
    print '    - {:25}: {:5}s'.format(
            'GaussianNB',
            round(time.time()-t0,2))
    
    # DecisionTreeClassifier
    dtree_tuner = ClassifierTuner(scaler=None, 
                                  selection=SelectKBest(),
                                  clf=DecisionTreeClassifier(
                                          random_state=24))
    dtree_tuner.scaler_params = dtree_scaler_params
    dtree_tuner.selection_params = dtree_selection_params
    dtree_tuner.clf_params = dtree_clf_params
    
    dtree_tuner.build_pipe()
    t0 = time.time()
    dtree_tuner.run_GridSearchCV(data, features, 20)
    print '    - {:25}: {:5}s'.format(
            'DecisionTreeClassifier',
            round(time.time()-t0,2))
    
    # RandomForestClassifier
    rforest_tuner = ClassifierTuner(scaler=None,
                                    selection=SelectKBest(),
                                    clf=RandomForestClassifier(
                                            random_state=24))
    rforest_tuner.scaler_params = rforest_scaler_params
    rforest_tuner.selection_params = rforest_selection_params
    rforest_tuner.clf_params = rforest_clf_params
    
    rforest_tuner.build_pipe()
    t0 = time.time()
    rforest_tuner.run_GridSearchCV(data, features, 20)
    print '    - {:25}: {:5}s'.format(
            'RandomForestClassifier',
            round(time.time()-t0,2))
    
    # AdaBoostClassifier
    aboost_tuner = ClassifierTuner(scaler=None,
                                   selection=SelectKBest(),
                                   clf=AdaBoostClassifier(
                                           random_state=24))
    aboost_tuner.scaler_params = aboost_scaler_params
    aboost_tuner.selection_params = aboost_selection_params
    aboost_tuner.clf_params = aboost_clf_params
    
    aboost_tuner.build_pipe()
    t0 = time.time()
    aboost_tuner.run_GridSearchCV(data, features, 20)
    print '    - {:25}: {:5}s'.format(
            'AdaBoostClassifier',
            round(time.time()-t0,2))
    
    return gnb_tuner, dtree_tuner, rforest_tuner, aboost_tuner

def tune_final_classifier(data, features):
    '''
    Create the final identifier classifer tuner using DecisionTreeClassifier
    and tune the classifier
    Args:
        data: the dataset
        features: selected features
    Returns:
        final identifier classifier tuner
    '''
    # Final Identifier is DecisionTreeClassifier using final features selected
    clf_tuner = ClassifierTuner(scaler=None,
                                selection=SelectKBest(),
                                clf=DecisionTreeClassifier(random_state=24))
    clf_tuner.scaler_params = dtree_scaler_params
    clf_tuner.selection_params = {
        'feature_selection__k': range(1,7)
        }
    clf_tuner.clf_params = dtree_clf_params
    
    clf_tuner.build_pipe()
    t0 = time.time()
    clf_tuner.run_GridSearchCV(data, features_list, 20)
    print '    - {:25}: {:5}s'.format(
            'Final Identifier',
            round(time.time()-t0,2))
    
    return clf_tuner    

def print_test_scores(final=False):
    '''
    Print test scores of the classifiers.
    Args:
        final: if True, print the final identifier scores. Else, print
               the scores of initial classifiers. Default: False.
    '''
    if final:
        print '************************************************'
        print '*** Final Classifier: DecisionTreeClassifier ***'
        print '************************************************'
        print ''
        t0 = time.time()
        clf_tuner.test_best_estimator(my_dataset, features_list)
        print 'Testing time: {}s'.format(round(time.time()- t0,2))
    else:
        print '******************'
        print '*** GaussianNB ***'
        print '******************'
        print ''
        t0 = time.time()
        gnb_tuner.test_best_estimator(my_dataset, initial_features)
        print 'Testing time: {}s'.format(round(time.time()- t0,2))
        print ''
        print '******************************'
        print '*** RandomForestClassifier ***'
        print '******************************'
        print ''
        t0 = time.time()
        rforest_tuner.test_best_estimator(my_dataset, initial_features)
        print 'Testing time: {}s'.format(round(time.time()- t0,2))
        print ''
        print '**************************'
        print '*** AdaBoostClassifier ***'
        print '**************************'
        print ''
        t0 = time.time()
        aboost_tuner.test_best_estimator(my_dataset, initial_features)
        print 'Testing time: {}s'.format(round(time.time()- t0,2))
        print ''
        print '******************************'
        print '*** DecisionTreeClassifier ***'
        print '******************************'
        print ''
        t0 = time.time()
        dtree_tuner.test_best_estimator(my_dataset, initial_features)
        print 'Testing time: {}s'.format(round(time.time()- t0,2))
        print ''
        
def get_selected_features(tuner, features):
    '''
    Get the SelectKBest selected features from the GridSearchCV best estimator.
    Args:
        tuner: the classifier tuner
        features: feature list used when tuning the classifier
    Returns:
        SelectKBest selected features
    '''
    kbest = tuner.grid_search.best_estimator_.named_steps['feature_selection']
    selected_features = [features[i+1] for i in kbest.get_support(indices=True)]
    return selected_features

def print_kbest_feature_scores(tuner, selected_features):
    '''
    Print the SelectKBest selected feature scores from the GridSearchCV best
    estimator.
    Args:
        tuner: the classifier tuner
        selected_features: SelectKBest selected features
    '''
    kbest = tuner.grid_search.best_estimator_.named_steps['feature_selection']
    kbest_scores = kbest.scores_
    kbest_pvalues = kbest.pvalues_
    selected_features_kbest = [(
            selected_features[i],
            kbest_scores[i],
            kbest_pvalues[i]) for i in range(len(selected_features))]
    selected_features_kbest = sorted(
            selected_features_kbest,
            key=lambda feature:float(feature[1]),
            reverse=True)
    
    print'{:24}{:5}  {:3}'.format('Feature Name', 'Score', 'P-value')
    for feature, score, pvalue in selected_features_kbest:
        print '{:24}{:05.2f}  {:3.2f}'.format(feature, score, pvalue)
    print ''
    
def print_dtree_feature_importances(tuner, selected_features):
    '''
    Print the DecisionTreeClassifier feature importances from 
    the GridSearchCV best estimator
    Args:
        tuner: the classifier tuner
        selected_features: SelectKBest selected features
    '''
    dtree = tuner.grid_search.best_estimator_.named_steps['clf']
    f_importances = dtree.feature_importances_
    f_importances_tupple = [(
            selected_features[i],
            f_importances[i]) for i in range(len(selected_features))]
    f_importances_tupple = sorted(
            f_importances_tupple,
            key=lambda feature:float(feature[1]),
            reverse=True)
    
    print'{:24}{:5}'.format('Feature Name', 'Feature Importance')
    for feature, f_importance in f_importances_tupple:
        print '{:24}{:5}'.format(feature, round(f_importance,2))
    print ''

def print_best_params(tuner):
    '''
    Print the best parameters from the GridSearchCV
    Arg:
        tuner: the classifier tuner
    '''
    for key in sorted(tuner.grid_search.best_params_.iterkeys(), 
                      reverse = True):
        print '{:25}: {}'.format(key, tuner.grid_search.best_params_[key])    

###############################################################################
# MAIN PROGRAM                                                                #
###############################################################################

my_dataset = dict(data_dict)
my_dataset = prepare_data(my_dataset)

print 'Preparing POI identifier...'
print 'Tuning different classifiers...'
print 'Tuning time:'

gnb_tuner, \
dtree_tuner, \
rforest_tuner, \
aboost_tuner = tune_classifiers(my_dataset, initial_features)
dtree_tuner_selected_features = get_selected_features(dtree_tuner, 
                                                      initial_features)

clf_tuner = tune_final_classifier(my_dataset, features_list)

clf_tuner_selected_features = get_selected_features(clf_tuner,
                                                    features_list)

print 'Finished loading in {}s'.format(round(time.time()-t1,2))
print ''

if __name__ == '__main__':   
    print '***********************************************************'
    print '*** POI IDENTIFIER USING ENRON FINANCIAL AND EMAIL DATA ***'
    print '*** By : Yasirah Z. Krueng                              ***'
    print '***********************************************************'
    print '' 
    
    print_summary()

    print 'Test Scores from Different Classifiers:'   
    # Please uncomment the code below to see initial test scores
    # from different classifiers.
    
    #print_test_scores()
    
    print '-----------------------------------------------------------------'
    print 'Since DecisitionTreeClassifier has the best F1 and recall scores,'
    print 'I chose to use this classifier in my POI identifier.'
    print ''
    print 'Here are feature scores and feature importances for'
    print 'initial DecisionTreeClassifier:'
    print ''
    print_kbest_feature_scores(dtree_tuner, 
                               dtree_tuner_selected_features)
    print_dtree_feature_importances(dtree_tuner,
                                    dtree_tuner_selected_features)
    
    print 'Based on feature importances, I selected only features that have'
    print 'importance value greater than 0 to be used in my final identifier:'
    print_data_features(features_list)
    print ''
    
    print 'Then, I tuned my identifier using this final features list,'
    print 'giving me the following final test score:'
    print ''
    print_test_scores(True)
    print ''
    
    print 'Here are the best parameters for my final identifier:'
    print ''
    print_best_params(clf_tuner)
    print ''
    
    print 'Here are the final feature scores and feature importances:'
    print ''
    print_kbest_feature_scores(clf_tuner,
                               clf_tuner_selected_features)
    print_dtree_feature_importances(clf_tuner,
                                    clf_tuner_selected_features)    
    
    clf = clf_tuner.grid_search.best_estimator_

    dump_classifier_and_data(clf, my_dataset, features_list)