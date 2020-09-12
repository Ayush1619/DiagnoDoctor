#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


dataframe = pd.read_csv("Downloads\diabetes.csv")


# In[3]:


dataframe.head()


# In[4]:


dataframe.shape


# In[5]:


dataframe.describe()


# In[6]:


sns.pairplot(dataframe[['Pregnancies','BloodPressure','BMI','Age','Glucose','Outcome']],hue='Outcome')


# In[8]:


train=dataframe.drop('Outcome',axis=1)
train.head()


# In[9]:


Outcome=dataframe.Outcome
Outcome.head()


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(train,Outcome, test_size=0.2, random_state=10)
print("X_train size  ==>", X_train.shape)
print("X_test size  ==>", X_test.shape)
print("Y_train size  ==>", Y_train.shape)
print("Y_test size  ==>", Y_test.shape)


# In[11]:


clf1=svm.SVC(kernel='linear')
clf1.fit(X_train,Y_train)
Y_pred=clf1.predict(X_test)


# In[12]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)


# In[15]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)


# In[19]:


Y_pred = classifier.predict(X_test)


# In[20]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[21]:


import pickle


# In[22]:


with open('model1_pickle','wb') as f:
    pickle.dump(clf1,f)


# In[23]:


with open('model1_pickle','rb') as f:
    mp = pickle.load(f)


# In[24]:


mp.predict(X_test)


# In[ ]:




