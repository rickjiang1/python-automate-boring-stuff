import pandas as pd
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
os.chdir('C:\PythonProject\Machine_Learning\dataset')
pokemon=pd.read_csv('Pokemon.csv')

pokemon.info()

#Add Type combo column
def type_combo(columns):
    type1=columns[0]
    type2=columns[1]
    if pd.isnull(type2):
        return type1
    else:
        return str(type1)+'-'+str(type2)
    
pokemon['type_combo']=pokemon[['Type 1','Type 2']].apply(type_combo,axis=1)

#Analysing

#how many legendary
Legendary=pokemon[pokemon['Legendary']==True]
print(pokemon.groupby('Legendary').count()['#'])
#for each type how many pokemons are legendary
print(pokemon[pokemon['Legendary']==True].groupby(['Type 1','Legendary']).count()['#'].sort_values())
#top 50 total, how many are the legendary
top_50=pokemon[['Name','Total','Legendary']].sort_values('Total',ascending=False).head(50)
#High Hp pokemon
High_HP_pokemon=pokemon[pokemon['HP']>pokemon['HP'].mean()*1.5]
#High Attack Pokemon
High_Attack_pokemon=pokemon[(pokemon['Attack']+pokemon['Sp. Atk'])>(pokemon['Attack']+pokemon['Sp. Atk']).mean()*1.5]
#High Defense pokemon
High_Defense_pokemon=pokemon[(pokemon['Defense']+pokemon['Sp. Def'])>(pokemon['Defense']+pokemon['Sp. Def']).mean()*1.5]
#mean of legandary pokemon's speed
High_Speed_pokemon=pokemon[pokemon['Speed']>pokemon['Speed'].mean()*1.5]
#correlation between overall
#sns.heatmap(pokemon.corr(),annot=pokemon.corr())


#get dummies & scaling
scaler=StandardScaler()

x=pokemon.drop(['#','Name','Type 2','Legendary'],axis=1)
x=pd.get_dummies(x,drop_first=True)
x=scaler.fit_transform(x)
y=pokemon['Legendary']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


#look at the score of each model-----------------------------------------------
'''
score=[]
classifier_LR=LogisticRegression()
classifier_LR.fit(X_train,y_train)
score.append('LR Score: '+str(classifier_LR.score(X_train,y_train)))

classifier_KNN=KNeighborsClassifier(n_neighbors=10)
classifier_KNN.fit(X_train,y_train)
score.append('KNN Score: '+str(classifier_KNN.score(X_train,y_train)))

classifier_SVC=SVC()
classifier_SVC.fit(X_train,y_train)
score.append('SVC Score: '+str(classifier_SVC.score(X_train,y_train)))

classifier_DT=DecisionTreeClassifier()
classifier_DT.fit(X_train,y_train)
score.append('DT Score: '+str(classifier_DT.score(X_train,y_train)))

classifier_RF=RandomForestClassifier(n_estimators=50)
classifier_RF.fit(X_train,y_train)
score.append('RF Score: '+str(classifier_DT.score(X_train,y_train)))
'''
#look at the score of each model-----------------------------------------------
'''
classifier_RF=RandomForestClassifier(n_estimators=100)
classifier_RF.fit(X_train,y_train)
pred_RF=classifier_RF.predict(X_test)
print(confusion_matrix(pred_RF,y_test))
print(classification_report(pred_RF,y_test))

'''
'''
[[144   5]
 [  1  10]]
             precision    recall  f1-score   support

      False       0.99      0.97      0.98       149
       True       0.67      0.91      0.77        11

avg / total       0.97      0.96      0.97       160
'''
'''
classifier_LR=LogisticRegression()
classifier_LR.fit(X_train,y_train)
pred_LR=classifier_LR.predict(X_test)
print(confusion_matrix(pred_LR,y_test))
print(classification_report(pred_LR,y_test))

[[133   4]
 [ 12  11]]
             precision    recall  f1-score   support

      False       0.92      0.97      0.94       137
       True       0.73      0.48      0.58        23

avg / total       0.89      0.90      0.89       160
'''

'''
classifier_SVC=SVC(kernel='rbf')
classifier_SVC.fit(X_train,y_train)
pred_SVC=classifier_SVC.predict(X_test)
print(confusion_matrix(pred_SVC,y_test))
print(classification_report(pred_SVC,y_test))
[[144  11]
 [  1   4]]
             precision    recall  f1-score   support

      False       0.99      0.93      0.96       155
       True       0.27      0.80      0.40         5

avg / total       0.97      0.93      0.94       160
'''

























