# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:53:52 2018

@author: ziweizh
"""
# load modules
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats.stats import spearmanr

# load data:
'''
df is used for 10-fold CV (change its path1 to designated location)
df1 is used for cross-prompt testing (change its path2 to designated location)
'''
path1 = 'PTJ1_ALL_LARC_2Feb_SMOTE.csv'
path2 = 'PTJ1_ALL_LARC_2Feb.csv'
df = pd.read_csv(path1)
df2 = pd.read_csv(path2)
# create variables
features= df.drop('Country', axis = 1)
features= features.drop('id', axis = 1)
features= features.drop('CEFR', axis = 1)
if "SMOTE" not in path1:
    features= features.drop('CEFR_NUM', axis = 1) # only applicable to non-oversampled data
feature_list = list(features.columns)
labels = df['CEFR'].astype('category')

x_test = df2.drop('Country', axis = 1)
x_test = x_test.drop('id', axis = 1)
x_test = x_test.drop('CEFR', axis = 1)
if "SMOTE" not in path2:
    x_test = x_test.drop('CEFR_NUM', axis = 1) # only applicable to non-oversampled data
y_test = df2['CEFR'].astype('category')

# define random forest classifier
rf = RandomForestClassifier(n_estimators = 50, random_state=1234)

# intrinsic evaluation: 10-fold CV
cross_val = cross_val_score(rf, features, labels, cv=StratifiedKFold(10,shuffle=True))
print sum(cross_val)/float(len(cross_val))
print("Accuracy: %0.2f (+/- %0.2f)" % (cross_val.mean(), cross_val.std() * 2)) 
# more detailed evaluation metrics
predicted=cross_val_predict(rf,features,labels,cv=StratifiedKFold(10,shuffle=True))
conf_mat = confusion_matrix(labels, predicted)
print conf_mat
cohen_kappa_score(labels, predicted)
cohen_kappa_score(labels, predicted, weights = "quadratic")
labels_fac=pd.factorize(labels)
predicted_fac = pd.factorize(predicted)
spearmanr(labels_fac[0],predicted_fac[0])
print classification_report(labels, predicted, target_names=['A2_0','B1_1','B1_2','B2_0'])

# extrinsic evaluation: cross-prompt evaluation (on a different prompt)
rf.fit(features, labels)
rf.score(x_test,y_test) 


# Feature Importance
'''
Feature selection (information gains for each feature from each tree are averaged)
impurity measure-based feature selelction has bias towards the feature that have more levels
multicollinearity: when multiple features are highly correlated with DV and each other, only one gets selected as important feature
'''
# 1). Variable importance
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
# simpler way to get feature importance
from tabulate import tabulate
headers = ["name", "score"]
values = sorted(zip(features.columns, rf.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))

# Performance of top 10 features: New random forest with only the 10 most important variables
features_small = df[['DC','wordtypes','MFCC13','repair','ndw','rttr','swordtokens','std_MFCC2','std_spectral_centroid','cttr','swordtypes','MLC']]
cross_val_small = cross_val_score(rf, features, labels, cv=10)
print sum(cross_val_small)/float(len(cross_val_small)) 


# # 2) Visualizing variable importance
import matplotlib.pyplot as plt
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(20,20))
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

'''
# Visualizing a single decision tree
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png') # this full tree may not be informative

# Limit depth of tree to 3 levels to generate a smaller tree
rf_small = RandomForestClassifier(n_estimators=10, random_state=10, max_depth=3,bootstrap=False)
rf_small.fit(features, labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True) #precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')
'''

# Constructing accuracy confidence intervals
import numpy
from pandas import read_csv
from matplotlib import pyplot
# load dataset
data = read_csv(path1)
del data['id']
del data['Country']
values = data.values
# configure bootstrap
n_iterations = 1000
n_size = int(len(data) * 0.80)
# run bootstrap
stats = list()
'''
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = numpy.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = RandomForestClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	print(score)
	stats.append(score)
'''
rf = RandomForestClassifier(n_estimators = 50)
for i in range(n_iterations):
	# evaluate model
	cross_val = cross_val_score(rf, features, labels, cv=StratifiedKFold(10,shuffle=True))
	score = np.mean(cross_val)
	print(score)
	stats.append(score)

# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, numpy.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, numpy.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

'''
# Hyper parameter tuning (# trees in the forest and max-depth) (on validation set) (to avoid overfitting)
# hyperparameters include the number of decision trees in the forest and the number of features considered by each tree when splitting a node
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
# Altogether, there are 2 * 12 * 2 * 3 * 3 * 10 = 4320 settings! 
# However, the benefit of a random search is that we are not trying every combination, but selecting at random to sample a wide range of values.
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(features, labels)
rf_random.best_params_ # view the best parameters from fitting the random search
# Evaluate random search (on different prompt): To determine if random search yielded a better model, we compare the base model with the best random search model.
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(features, labels)
base_accuracy = evaluate(base_model, x_test, y_test)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# Grid search with cross-validation
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model: instead of sampling randomly from a distribution, evaluates all combinations we define
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(features, labels)
grid_search.best_params_
{'bootstrap': True,
 'max_depth': 80,
 'max_features': 3,
 'min_samples_leaf': 5,
 'min_samples_split': 12,
 'n_estimators': 100}
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, x_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
# Training curves:
# Number of trees:
# Grid with only the number of trees changed
tree_grid = {'n_estimators': [int(x) for x in np.linspace(1, 301, 30)]}

# Create the grid search model and fit to the training data
tree_grid_search = GridSearchCV(final_model, param_grid=tree_grid, verbose = 2, n_jobs=-1, cv = 3,
                                scoring = 'neg_mean_absolute_error')
tree_grid_search.fit(train_features, train_labels)
tree_grid_search.cv_results_

import matplotlib.pyplot as plt
def plot_results(model, param = 'n_estimators', name = 'Num Trees'):
    param_name = 'param_%s' % param

    # Extract information from the cross validation model
    train_scores = model.cv_results_['mean_train_score']
    test_scores = model.cv_results_['mean_test_score']
    #train_time = model.cv_results_['mean_fit_time']
    param_values = list(model.cv_results_[param_name])
    
    # Plot the scores over the parameter
    #plt.subplots(1, 2, figsize=(10, 6))
    #plt.subplot(121)
    plt.plot(param_values, train_scores, 'bo-', label = 'train')
    plt.plot(param_values, test_scores, 'go-', label = 'test')
    plt.ylim(ymin = 0, ymax = 1.2)
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('accuracy')
    plt.title('Score vs %s' % name)
    
    plt.subplot(122)
    plt.plot(param_values, train_time, 'ro-')
    plt.ylim(ymin = 0.0, ymax = 2.0)
    plt.xlabel(name)
    plt.ylabel('Train Time (sec)')
    plt.title('Training Time vs %s' % name)
    
    
    plt.tight_layout(pad = 4)
plot_results(tree_grid_search)
# Number of Features at Each Split
# Define a grid over only the maximum number of features
feature_grid = {'max_features': list(range(1, train_features.shape[1] + 1))}
# Create the grid search and fit on the training data
feature_grid_search = GridSearchCV(final_model, param_grid=feature_grid, cv = 3, n_jobs=-1, verbose= 2,
                                  scoring = 'neg_mean_absolute_error')
feature_grid_search.fit(train_features, train_labels)
plot_results(feature_grid_search, param='max_features', name = 'Max Features')
'''

