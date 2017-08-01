#!/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree, cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as pp
from sklearn import linear_model
import sys



############## Fill missing ages using RandomForestClassifier ##################
def addMissingAgesUsingRandomForest(df, rfr=None):
  df_age = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

  known_age = df_age[df_age.Age.notnull()].as_matrix() 
  unknown_age = df_age[df_age.Age.isnull()].as_matrix() 

  y = known_age[:, 0] 
  X = known_age[:, 1:]

  if not rfr:  
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

  df.loc[(df.Age.isnull()), 'Age'] = rfr.predict(unknown_age[:, 1:])

  return df, rfr



########################### Data scrubbing #####################################
def dataSrubbing(df, rfr=None, age_scale_param=None, fare_scale_param=None):

  # Replace the NA values in the *Age* column with Random Forest Regressor.
  df, rfr  = addMissingAgesUsingRandomForest(df, rfr)

  df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Y'
  df.loc[(df.Cabin.isnull()), 'Cabin'] = 'N'

  dummies_cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
  dummies_embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
  dummies_sex = pd.get_dummies(df['Sex'], prefix='Sex')
  dummies_pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

  df = pd.concat([df, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass], axis=1)

  # Scaling
  scaler = pp.StandardScaler()
  if not age_scale_param: age_scale_param = scaler.fit(df['Age'])
  df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
  if not fare_scale_param: fare_scale_param = scaler.fit(df['Fare'])
  df['Fare_scaled'] = scaler.fit_transform(df['Fare'], age_scale_param)

  df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare'], axis=1, inplace=True)

  return df, rfr,  age_scale_param, fare_scale_param



########################### Performance ##################################
def measurePerformance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred=clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}\n".format(metrics.accuracy_score(y,y_pred)))

    if show_classification_report:
        print("Classification report:\n")
        print(metrics.classification_report(y,y_pred))

    if show_confusion_matrix:
        print("Confusion matrix:\n")
        print(metrics.confusion_matrix(y,y_pred))


########################### Logistic Regression ##################################
train = pd.read_csv('train.csv')
#print titanic.head()
#print titanic.describe()
#print titanic.info()
test = pd.read_csv('test.csv')
test.loc[(test.Fare.isnull()), 'Fare'] = 0
passenger_id = test['PassengerId']

df_train, rfr, age_scale_param, fare_scale_param = dataSrubbing(train) 
df_test, rfr, age_scale_param, fare_scale_param = dataSrubbing(test, rfr, age_scale_param, fare_scale_param) 
print df_test.head()


df_train = df_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
np_train = df_train.as_matrix()
y = np_train[:,0]
X = np_train[:,1:]

lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
lr.fit(X,y)


df_test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = lr.predict(df_test)
result = pd.DataFrame({'PassengerId':passenger_id.as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv('test_prediction.csv', index=False)
