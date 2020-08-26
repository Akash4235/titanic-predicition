#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv(r"C:\Users\aakas\Documents\competetion\titanic\train.csv")


# In[3]:


# Dropping the cabin vaiables as 80% of the data is missing.
# Embarked has 2 values missing so will drop these 2 rows.
print(train_data.isnull().sum())


# In[5]:


train_data.drop('Cabin',axis=1,inplace=True)
# Correlation of data
plt.figure(figsize=(8,6))
sns.heatmap(train_data.corr(),annot=True,cmap='rainbow',linewidths=.02,linecolor='black')

# As the age column is correlated to Pclass so we will use it for filling the values of Age.
# It is a fair relation as wealthier people are aged.
# So we will use Pclass column to fill the Age column. Boxplot chart shows further explanation.

plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass',y='Age',data=train_data)

# Function to fill the missing values in Age column.
def age_fill(i):
    Pclass = i[0]
    Age= i[0]
    
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 23
    else:
        return Age
    
# lets apply function to the data.
train_data['Age'] = train_data[['Age','Pclass']].apply(age_fill,axis=1)


# In[6]:


# Getting Dummy Variables/One-Hot-Encoding for the data.
sex = pd.get_dummies(train_data['Sex'],drop_first=True)
embark = pd.get_dummies(train_data['Embarked'],drop_first = True)
train_data = pd.concat([sex,embark,train_data],axis=1)

# We don't need column Name because the Sex and Age of the passengers are already given 
# so we don't need to collect Titles like Mr., Mrs., Miss.
# Also, we don't need the columns Fare and Ticket
train_data.drop(['Sex','Embarked','Name','Fare','Ticket'],axis=1,inplace=True)


# In[7]:


# Training model using Logistic Regression.
from sklearn.model_selection import train_test_split

X = train_data.drop('Survived',axis=1)
y = train_data['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)


# In[8]:


from sklearn.linear_model import LogisticRegression

logR = LogisticRegression(max_iter=5000)
logR.fit(X_train,y_train)


# In[9]:


predictions = logR.predict(X_test)


# In[10]:


from sklearn.metrics import accuracy_score, classification_report
print('Accuracy Score = ', accuracy_score(y_test,predictions)*100)
print('\n')
print(classification_report(y_test,predictions))
# Accuracy Score of 81%

