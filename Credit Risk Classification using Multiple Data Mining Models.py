import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('./Dataset/GermanCredit.csv')
df.head(10)
df.Class.value_counts()
print('There are 700 applicants with good risk and 300 with bad risk.'
      'In classification problems imbalanced datasets where one class is underrepresented' 
      'makes it difficult for a model to recognize the underrepresented class.' 
      'In this case, the smaller class still has a good number of samples that it would not be a problem.')
target = df.Class
data = df.drop(['Class'], axis=1)

from sklearn.model_selection import ShuffleSplit

# split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30)


# perform hyperperameter tuning with cross validation and grid search to find optimal values
decision_tree_classifier = DecisionTreeClassifier(random_state=0)
parameter_grid = {'max_depth': np.arange(3, 15),
                  'max_features': np.arange(1, 10)}

cross_validation = ShuffleSplit(n_splits = 10, test_size = 0.30)

grid_search = GridSearchCV(decision_tree_classifier, param_grid = parameter_grid,
                          cv = cross_validation)

grid_search.fit(X_train.values, y_train.values)

print ("Best Score: {}".format(grid_search.best_score_))
print ("Best params: {}".format(grid_search.best_params_))

optimized_clf = grid_search.best_estimator_
optimized_clf.predict(X_test)



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# use optimal parameters to predict credit risk
y_true, y_pred = y_test, optimized_clf.predict(X_test)
print(classification_report(y_true, y_pred))

mat = confusion_matrix(y_true, y_pred, labels=[1,0])

sns.heatmap(mat, square=True, annot=True, fmt='d', linewidths=.5, cmap="YlGnBu")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

print("Accuracy on training set: {:.3f}".format(optimized_clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(optimized_clf.score(X_test, y_test)))

from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *


viz = dtreeviz(optimized_clf,
               X_train.values,
               y_train.values,
               target_name='Credit',
               feature_names = X_train.columns,
                class_names=["Bad","Good"])
              
viz  

# Now using Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# hyperparameters

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
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

random_classifier = RandomForestClassifier()

cross_validation = ShuffleSplit(n_splits = 3, test_size = 0.30)

grid_search2 = GridSearchCV(random_classifier, 
                            param_grid = random_grid,
                            cv = cross_validation, 
                            verbose=1,
                            n_jobs = -1)

grid_search2.fit(X_train.values, y_train.values)

print ("Best Score: {}".format(grid_search2.best_score_))
print ("Best params: {}".format(grid_search2.best_params_))

grid_search2.best_params_
optimal_rf = grid_search2.best_estimator_
optimal_rf.predict(X_test)

y_true, y_pred = y_test, optimal_rf.predict(X_test)
print(classification_report(y_true, y_pred))

mat = confusion_matrix(y_true, y_pred, labels=[1,0])

sns.heatmap(mat, square=True, annot=True, fmt='d', linewidths=.5, cmap="YlGnBu")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

print("Accuracy on training set: {:.3f}".format(optimal_rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(optimal_rf.score(X_test, y_test)))

# Now using XGBoost
from xgboost.sklearn import XGBClassifier

# A parameter grid for XGBoost
xgb_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


xgb = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=10)


cross_validation = ShuffleSplit(n_splits = 3, test_size = 0.30)

grid_search3 = GridSearchCV(xgb, 
                            param_grid = xgb_params,
                            cv = cross_validation, 
                            verbose=1,
                            n_jobs = -1)


grid_search3.fit(X_train.values, y_train.values)

print ("Best Score: {}".format(grid_search3.best_score_))
print ("Best params: {}".format(grid_search3.best_params_))

optimal_xgb = grid_search3.best_estimator_
optimal_xgb
optimal_xgb = grid_search3.best_estimator_
y_true, y_pred = y_test, optimal_xgb.predict(X_test.values)
print(classification_report(y_true, y_pred))

mat = confusion_matrix(y_true, y_pred, labels=[1,0])

sns.heatmap(mat, square=True, annot=True, fmt='d', linewidths=.5, cmap="YlGnBu")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

print("Accuracy on training set: {:.3f}".format(optimal_xgb.score(X_train.values, y_train.values)))
print("Accuracy on test set: {:.3f}".format(optimal_xgb.score(X_test.values, y_test.values)))

# Catboost

# generate list of indices for all categorical columns
bool_cols = [col for col in X_train if 
               X_train[col].dropna().value_counts().index.isin([0,1]).all()]

cat_features = [X_train.columns.get_loc(c) for c in bool_cols if c in X_train]

import catboost as cb

cat = cb.CatBoostClassifier()
cat_params = {    'depth'         : [6,8,10],
                  'learning_rate' : [0.01, 0.05, 0.1],
                  'iterations'    : [30, 50, 100]
                 }
    

cross_validation = ShuffleSplit(n_splits = 3, test_size = 0.30)

grid_search4 = GridSearchCV(
                            cat,
                            param_grid = cat_params,
                            cv = cross_validation, 
                            verbose=1,
                            n_jobs = -1)


grid_search4.fit(X_train.values, y_train.values, cat_features= cat_features)

print ("Best Score: {}".format(grid_search4.best_score_))
print ("Best params: {}".format(grid_search4.best_params_))
optimal_cat = grid_search4.best_estimator_
y_true, y_pred = y_test, optimal_cat.predict(X_test.values)
print(classification_report(y_true, y_pred))

mat = confusion_matrix(y_true, y_pred, labels=[1,0])

sns.heatmap(mat, square=True, annot=True, fmt='d', linewidths=.5, cmap="YlGnBu")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

print("Accuracy on training set: {:.3f}".format(optimal_cat.score(X_train.values, y_train.values)))
print("Accuracy on test set: {:.3f}".format(optimal_cat.score(X_test.values, y_test.values)))

# LightGMB Classifier
import lightgbm as lgb


lg = lgb.LGBMClassifier(silent=False)
lg_params = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }

grid_search5 = GridSearchCV(lg, 
                            param_grid = lg_params,
                            cv = cross_validation, 
                            verbose=1,
                            n_jobs = -1)


grid_search5.fit(X_train.values, y_train.values)

print ("Best Score: {}".format(grid_search5.best_score_))
print ("Best params: {}".format(grid_search5.best_params_))
optimal_lg = grid_search5.best_estimator_
y_true, y_pred = y_test, optimal_lg.predict(X_test.values)
print(classification_report(y_true, y_pred))

mat = confusion_matrix(y_true, y_pred, labels=[1,0])

sns.heatmap(mat, square=True, annot=True, fmt='d', linewidths=.5, cmap="YlGnBu")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

print("Accuracy on training set: {:.3f}".format(optimal_lg.score(X_train.values, y_train.values)))
print("Accuracy on test set: {:.3f}".format(optimal_lg.score(X_test.values, y_test.values)))

models = [optimized_clf, optimal_rf, optimal_xgb, optimal_cat, optimal_lg]
scores = []
model_names = ['decision_tree','random_forest','xgboost','catboost','lightgbm']

for model in models:
  scores.append(model.score(X_test.values, y_test.values))
  
pd.DataFrame({
    'model': model_names,
    'accuracy': scores
})













