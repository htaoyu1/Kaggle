#!/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree, cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier


titanic = pd.read_csv('train.csv')
#print titanic.head()
#print titanic.describe()


########################### Data scrubbing #####################################
def dataSrubbing(df):

  # Replace the NA values in the *Age* column with median value.
  df['Age'] = df['Age'].fillna(df['Age'].median())

  # Replace *Sex* values with numbers: 0 (male), 1(female).
  df.loc[df['Sex'] == 'male', 'Sex'] = 0
  df.loc[df['Sex'] == 'female', 'Sex'] = 1

  # Replace *Embarked* values with numbers: 0(S), 1(C), 2(Q).
  #print (df['Embarked'].unique())  # uniqe values for *Embarked*
  df['Embarked'] = df['Embarked'].fillna('S')
  df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
  df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
  df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2
  return df


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


########################### Random Forest  ##################################
titanic = dataSrubbing(titanic) 
y = titanic.Survived
age = titanic.Age
pclass = pd.get_dummies(titanic.Pclass, prefix='Pclass')
sex = titanic['Sex']
X = pd.concat([pclass, age, sex], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


clf2 = RandomForestClassifier(n_estimators=1000,random_state=33)
clf2 = clf2.fit(X_train,y_train)
measurePerformance(X_test,y_test,clf2, show_classification_report=True, show_confusion_matrix=True)


scores2 = cross_validation.cross_val_score(clf2,X, y, cv=10)
print("Cross Validation Score: {0:.3f}\n".format(scores2.mean()))
