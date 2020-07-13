# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:15:53 2020

@author: 16487
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cbt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import warnings

def newage (cols):
    title=cols[0]
    Sex=cols[1]
    Age=cols[2]
    if pd.isnull(Age):
        if title=='High' and Sex=="male":
            return 4.57
        elif title=='Low' and Sex=='male':
            return 32.65
        elif title=='Middle' and Sex=='male': 
            return 46.22
        elif title=='Middle' and Sex=='female':
            return 49.00
        elif title=='Top' and Sex=='female':
            return 27.83
        elif title=='Top' and Sex=='male':
            return 49.00
        else:
            return 42.33
    else:
        return Age 

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
test2=pd.read_csv("test.csv")
titanic=pd.concat([train, test], sort=False)  #合并训练集和测试集
len_train=train.shape[0]  #训练数据长度
#print(titanic)
#print(titanic.isnull().sum()[titanic.isnull().sum()>0])  #查看缺失数据

#补充缺失数据
train.Fare=train.Fare.fillna(train.Fare.mean())
test.Fare=test.Fare.fillna(train.Fare.mean())

train.Embarked=train.Embarked.fillna(train.Embarked.mode()[0])
test.Embarked=test.Embarked.fillna(train.Embarked.mode()[0])

train.Cabin=train.Cabin.fillna("Unknow")
test.Cabin=test.Cabin.fillna("Unknow")
train.Cabin = train.Cabin.map(lambda x: x[0])  #对于Cabin用首字母代替
test.Cabin = test.Cabin.map(lambda x: x[0])  #对于Cabin用首字母代替

train['title']=train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
test['title']=test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
#plt.figure(figsize=[20,10])  
#sns.barplot('title','Survived',data=train)  #做图查看各title生存比例
#sns.countplot(x='title', hue='Survived', data=train)
#print(train['title'].value_counts())  #查看title中各名称的个数
#sns.barplot('title','Age',data=train)
#print(train.groupby(['title']).Age.median()) #查看年龄中位数
#print(train.groupby(['title']).Age.mean()) #查看年龄平均值
#print(train.groupby(['title','Sex']).Age.mean()) #查看年龄平均值
#print(train['title'].value_counts())  #查看title中各名称的个数

newtitles={
    "Capt":       "Low",  #船长
    "Col":        "Middle",  
    "Major":      "Middle",
    "Jonkheer":   "Low",  #乡绅
    "Don":        "Low",  #教授
    "Sir" :       "Top",       #先生
    "Dr":         "Middle",  #医生
    "Rev":        "Low",  #教师
    "the Countess":"Top", #女伯爵
    "Dona":       "Top",  #夫人
    "Mme":        "Top",      #小姐
    "Mlle":       "Top",     #小姐
    "Ms":         "Top",
    "Mr" :        "Low",
    "Mrs" :       "Top",
    "Miss" :      "Top",
    "Master" :    "High",
    "Lady" :      "Top" } #女士      
train['title']=train.title.map(newtitles)
test['title']=test.title.map(newtitles)
#print(train.groupby(['title','Sex']).Age.mean()) #查看年龄平均值

train.Age=train[['title','Sex','Age']].apply(newage, axis=1)
test.Age=test[['title','Sex','Age']].apply(newage, axis=1)

train['FamilySize'] = train.Parch + train.SibSp + 1
test['FamilySize'] = test.Parch + test.SibSp + 1

titanic=pd.concat([train, test], sort=False)  #合并训练集和测试集
len_train=train.shape[0]  #训练数据长度
#print(titanic.isnull().sum()[titanic.isnull().sum()>0])  #查看缺失数据
titanic.Sex = titanic.Sex.map({"male": 0, "female":1})  #对于Sex用0，1代替
print(titanic)

#做图观察特征
'''''
warnings.filterwarnings(action="ignore")
plt.figure(figsize=[12,10])
plt.subplot(3,3,1)  #将显示区域分成3x3，该图位于1区域
sns.barplot('Pclass','Survived',data=train)
plt.subplot(3,3,2)
sns.barplot('FamilySize','Survived',data=train)
plt.subplot(3,3,2)
sns.barplot('Cabin','Fare',data=train)
plt.subplot(3,3,4)
sns.barplot('Sex','Survived',data=train)
plt.subplot(3,3,5)
sns.barplot('title','Survived',data=train)
plt.subplot(3,3,6)
sns.barplot('Cabin','Survived',data=train)
plt.subplot(3,3,7)
sns.barplot('Embarked','Survived',data=train)
plt.subplot(3,3,8)
sns.distplot(train[train.Survived==1].Age, color='green', kde=False)
sns.distplot(train[train.Survived==0].Age, color='orange', kde=False)
plt.subplot(3,3,9)
sns.distplot(train[train.Survived==1].Fare, color='green', kde=False)
sns.distplot(train[train.Survived==0].Fare, color='orange', kde=False)


#Sex
fig, axarr = plt.subplots(1, 2, figsize=(12,6))
a = sns.countplot(train['Sex'], ax=axarr[0]).set_title('Passengers count by sex')
axarr[1].set_title('Survival rate by sex')
b = sns.barplot(x='Sex', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
#Pclass
fig, axarr = plt.subplots(1,2,figsize=(12,6))
a = sns.countplot(x='Pclass', hue='Survived', data=train, 
                  ax=axarr[0]).set_title('Survivors and deads count by class')
axarr[1].set_title('Survival rate by class')
b = sns.barplot(x='Pclass', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
#Age
fig, axarr = plt.subplots(1,2,figsize=(12,6))
axarr[0].set_title('Age distribution')
f = sns.distplot(train['Age'], color='red', bins=40, ax=axarr[0])
axarr[1].set_title('Age distribution for the two subpopulations')
g = sns.kdeplot(train['Age'].loc[train['Survived'] == 1], 
                shade= True, ax=axarr[1], label='Survived').set_xlabel('Age')
g = sns.kdeplot(train['Age'].loc[train['Survived'] == 0], 
                shade=True, ax=axarr[1], label='Not Survived')
#Fare
fig, axarr = plt.subplots(1,2,figsize=(12,6))
f = sns.distplot(train.Fare, color='g', ax=axarr[0]).set_title('Fare distribution')
fare_ranges = pd.qcut(train.Fare, 4, labels = ['Low', 'Mid', 'High', 'Very high'])
axarr[1].set_title('Survival rate by fare category')
g = sns.barplot(x=fare_ranges, y=train.Survived, ax=axarr[1]).set_ylabel('Survival rate')
#Embarked
fig, axarr = plt.subplots(1,2,figsize=(12,6))
sns.countplot(train['Embarked'], ax=axarr[0]).set_title('Passengers count by boarding point')
p = sns.countplot(x = 'Embarked', hue = 'Survived', data = train, 
                  ax=axarr[1]).set_title('Survivors and deads count by boarding point')
fig, axarr = plt.subplots(1,figsize=(6,6))
g = sns.countplot(data=train, x='Embarked', hue='Pclass').set_title('Pclass count by embarking point')
'''


#划分数据，用模型进行预测


train.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)

titanic=pd.concat([train, test], sort=False)

titanic=pd.get_dummies(titanic)  #将字符型特征变为int型(增加特征)

train=titanic[:len_train]
test=titanic[len_train:]

train.Survived=train.Survived.astype('int')

xtrain=train.drop("Survived",axis=1)
ytrain=train['Survived']
xtest=test.drop("Survived", axis=1)

print(titanic.columns.values)


#随机森林
RF=RandomForestClassifier(random_state=None)
PRF=dict(max_depth = [10],     
         min_samples_split = [7], 
         min_samples_leaf = [4],     
         n_estimators = [60]) 
GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=10)
scores_rf=cross_val_score(GSRF,xtrain,ytrain,scoring='accuracy',cv=10)
print("rf : ",np.mean(scores_rf))

#svm
svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)
scores_svm=cross_val_score(GSSVM, xtrain.astype(float), ytrain,scoring='accuracy', cv=5)
print("svm : ",np.mean(scores_svm))

#catboost
CBT=cbt.CatBoostClassifier(iterations=1000,learning_rate=0.01,verbose=300,early_stopping_rounds=200,loss_function='MultiClass')
scores_cbt=cross_val_score(CBT,xtrain,ytrain,scoring='accuracy',cv=3)
print("cbt : ",np.mean(scores_cbt))

#输出到文件
model=GSRF.fit(xtrain, ytrain)
#print("Optimal params: {}".format(model.best_estimator_))
pred=model.predict(xtest)
print(pred)
output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})
output.to_csv('submission_rf.csv', index=False)

model=GSSVM.fit(xtrain, ytrain)
#print("Optimal params: {}".format(model.best_estimator_))
pred=model.predict(xtest)
output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})
output.to_csv('submission_svm.csv', index=False)

model=CBT.fit(xtrain, ytrain)
pred=model.predict(xtest)
result=pd.DataFrame(pred.astype(int))
output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':result[0]})
output.to_csv('submission_cbt.csv', index=False)





