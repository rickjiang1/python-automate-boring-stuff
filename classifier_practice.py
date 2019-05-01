import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

os.chdir('C:\PythonProject\Machine_Learning\dataset')
titanic=pd.read_csv('titanic_train.csv')
titanic.info()
print(titanic.columns)

#Data cleaning---------------------------------------------------------------------

#missing data
def fillage (column):
    Age=column[0]
    Pclass=column[1]

    if pd.isnull(Age):
        if Pclass==1:
            return int(titanic[titanic['Pclass']==1]['Age'].mean())
        elif Pclass==2:
            return int(titanic[titanic['Pclass']==2]['Age'].mean())
        else:
            return int(titanic[titanic['Pclass']==3]['Age'].mean())
    else:
        return Age




titanic['Age']=titanic[['Age','Pclass']].apply(fillage,axis=1)
titanic=titanic[titanic['Embarked'].isna()==False]

#CreateAgeBin
titanic['Agebin'] = pd.cut(titanic['Age'], 5)
#replace Age to bin value
def replace_age(Age):
    if int(Age)<16:
        return 0
    elif int(Age)>16 & int(Age)<32:
        return 1
    elif int(Age) >32 & int(Age)<64:
        return 2
    else:
        return 3

titanic['Age']=titanic['Age'].apply(replace_age)
title=[i for i in titanic['Name'].apply(lambda x:x.split('.')[0].split(',')[1])]
titanic['Title']=title
titanic['Title']=titanic['Title'].replace([' Capt',' Col',' Don',' Dr',
       ' Jonkheer',' Lady',' Major',' Mlle',' Mme',' Ms',' Rev',' Sir',' the Countess'],'Rare')
titanic['Title']=titanic['Title'].str.strip()

#Data cleaning---------------------------------------------------------------------



#Data analysing---------------------------------------------------------------------
'''
print(titanic.describe())
#how many people survived and how many people died?
print(titanic.groupby('Survived').count()['PassengerId'])
#for each class, how many passenger survived?
print(titanic.groupby(['Pclass','Survived']).count()['PassengerId'])
#for each class, the percentage of people survived
print(titanic[['Pclass','Survived']].groupby(['Pclass']).mean())
#how many survived people has Sib
print(titanic.groupby(['SibSp','Survived']).count()['PassengerId'])
#avg fare of each class
print(titanic.groupby('Pclass').mean()['Fare'])
#Age
g = sns.FacetGrid(titanic, col='Survived')
g.map(plt.hist, 'Age', bins=20)

g = sns.FacetGrid(titanic, col='Survived')
g.map(sns.countplot, 'Pclass')


print(titanic[['Agebin', 'Survived']].groupby(['Agebin'], as_index=False).mean().sort_values(by='Agebin', ascending=True))

'''


#Data analysing---------------------------------------------------------------------



#Dummies Variables
x=titanic.drop(['Survived','PassengerId','Name','Ticket','Cabin','Agebin'],axis=1)
x=pd.get_dummies(x,drop_first=True)
y=titanic['Survived']

#Scaling

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)



#CREATE MODELS-----------------------------------------------------------------------
#Logistic Regression

classifier_LR=LogisticRegression ()
classifier_LR.fit(X_train,y_train)
pred_LR=classifier_LR.predict(X_test)
pred_LR_percentage=classifier_LR.predict_proba(X_test)
print(pred_LR_percentage)
#look at the confidence score
acc_log_LR = round(classifier_LR.score(X_train, y_train) * 100, 2)

'''
[[138  25]
 [ 29  75]]
              precision    recall  f1-score   support

           0       0.83      0.85      0.84       163
           1       0.75      0.72      0.74       104

   micro avg       0.80      0.80      0.80       267
   macro avg       0.79      0.78      0.79       267
weighted avg       0.80      0.80      0.80       267



After tunning:
[[143  25]
 [ 24  75]]
             precision    recall  f1-score   support

          0       0.86      0.85      0.85       168
          1       0.75      0.76      0.75        99

avg / total       0.82      0.82      0.82       267
'''
'''

#KNN

error_rate=[]
for i in range(1,40):
    classifier_KNN=KNeighborsClassifier(n_neighbors=i)
    classifier_KNN.fit(X_train,y_train)
    pred_KNN=classifier_KNN.predict(X_test)
    error_rate.append(np.sum(pred_KNN!=y_test))

plt.plot(range(1,40),error_rate)

classifier_KNN=KNeighborsClassifier(n_neighbors=14)
classifier_KNN.fit(X_train,y_train)
pred_KNN=classifier_KNN.predict(X_test)

#look at the confidence score
acc_log_KNN = round(classifier_KNN.score(X_train, y_train) * 100, 2)

k=5
[[142  28]
 [ 25  72]]
              precision    recall  f1-score   support

           0       0.85      0.84      0.84       170
           1       0.72      0.74      0.73        97

   micro avg       0.80      0.80      0.80       267
   macro avg       0.79      0.79      0.79       267
weighted avg       0.80      0.80      0.80       267



K=14
[[151  35]
 [ 16  65]]
              precision    recall  f1-score   support

           0       0.90      0.81      0.86       186
           1       0.65      0.80      0.72        81

   micro avg       0.81      0.81      0.81       267
   macro avg       0.78      0.81      0.79       267
weighted avg       0.83      0.81      0.81       267

After Tunning:

[[144  32]
 [ 23  68]]
             precision    recall  f1-score   support

          0       0.86      0.82      0.84       176
          1       0.68      0.75      0.71        91

avg / total       0.80      0.79      0.80       267
'''
#SVC
'''
classifier_SVC=SVC(kernel='rbf')
classifier_SVC.fit(X_train,y_train)
pred_SVC=classifier_SVC.predict(X_test)

#look at the confidence score
acc_log_SVC_rbf = round(classifier_SVC.score(X_train, y_train) * 100, 2)

kernel='linear'
[[139  28]
 [ 28  72]]
              precision    recall  f1-score   support

           0       0.83      0.83      0.83       167
           1       0.72      0.72      0.72       100

   micro avg       0.79      0.79      0.79       267
   macro avg       0.78      0.78      0.78       267
weighted avg       0.79      0.79      0.79       267


kernel='rbf'
[[151  35]
 [ 16  65]]
              precision    recall  f1-score   support

           0       0.90      0.81      0.86       186
           1       0.65      0.80      0.72        81

   micro avg       0.81      0.81      0.81       267
   macro avg       0.78      0.81      0.79       267
weighted avg       0.83      0.81      0.81       267

After Tunning
[[147  30]
 [ 20  70]]
             precision    recall  f1-score   support

          0       0.88      0.83      0.85       177
          1       0.70      0.78      0.74        90

avg / total       0.82      0.81      0.81       267
'''

#Decision Tree
'''
classifier_DT=DecisionTreeClassifier()
classifier_DT.fit(X_train,y_train)
pred_DT=classifier_DT.predict(X_test)

#look at the confidence score
acc_log_DT= round(classifier_DT.score(X_train, y_train) * 100, 2)

[[135  30]
 [ 32  70]]
              precision    recall  f1-score   support

           0       0.81      0.82      0.81       165
           1       0.70      0.69      0.69       102

   micro avg       0.77      0.77      0.77       267
   macro avg       0.75      0.75      0.75       267
weighted avg       0.77      0.77      0.77       267


After Tuning:
[[144  30]
 [ 23  70]]
             precision    recall  f1-score   support

          0       0.86      0.83      0.84       174
          1       0.70      0.75      0.73        93

avg / total       0.81      0.80      0.80       267
'''
#RandomForest
'''
classifier_RF= RandomForestClassifier(n_estimators=100)
classifier_RF.fit(X_train,y_train)
pred_RF=classifier_RF.predict(X_test)

#look at the confidence score
acc_log_RF = round(classifier_RF.score(X_train, y_train) * 100, 2)

ntrees=5
[[137  37]
 [ 30  63]]
              precision    recall  f1-score   support

           0       0.82      0.79      0.80       174
           1       0.63      0.68      0.65        93

   micro avg       0.75      0.75      0.75       267
   macro avg       0.73      0.73      0.73       267
weighted avg       0.75      0.75      0.75       267


ntrees=100
[[138  34]
 [ 29  66]]
              precision    recall  f1-score   support

           0       0.83      0.80      0.81       172
           1       0.66      0.69      0.68        95

   micro avg       0.76      0.76      0.76       267
   macro avg       0.74      0.75      0.75       267
weighted avg       0.77      0.76      0.77       267

After Tuning:
    [[141  32]
 [ 26  68]]
             precision    recall  f1-score   support

          0       0.84      0.82      0.83       173
          1       0.68      0.72      0.70        94

avg / total       0.79      0.78      0.78       267
'''

#Naive Bay

#CREATE MODELS-----------------------------------------------------------------------


#Evaluation-confusion matricx
'''
print(confusion_matrix(pred_RF,y_test))
print(classification_report(pred_RF,y_test))
'''
#Evaluation-CAP Curve
