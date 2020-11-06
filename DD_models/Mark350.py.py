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


dataframe = pd.read_csv("Downloads\Mark1.csv")


# In[3]:


dataframe.head()


# In[4]:


dataframe.shape


# In[5]:


dataframe.describe()


# In[6]:


sns.pairplot(dataframe[['stomach_pain','weight_loss','cough','chest_pain','continuous_feel_of_urine','outcome']],hue='outcome')


# In[7]:


train=dataframe.drop('outcome',axis=1)
train.head()


# In[8]:


outcome=dataframe.outcome
outcome.head()


# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(train,outcome, test_size=0.2, random_state=0)
print("X_train size  ==>", X_train.shape)
print("X_test size  ==>", X_test.shape)
print("Y_train size  ==>", Y_train.shape)
print("Y_test size  ==>", Y_test.shape)


# In[10]:


clf1=svm.SVC(kernel='linear')
clf1.fit(X_train,Y_train)
Y_pred=clf1.predict(X_test)


# In[11]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)


# In[14]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)


# In[17]:


Y_pred = classifier.predict(X_test)


# In[18]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[19]:


import pickle


# In[20]:


with open('mark350_pickle','wb') as f:
    pickle.dump(clf1,f)


# In[21]:


with open('mark350_pickle','rb') as f:
    mp = pickle.load(f)


# In[22]:


mp.predict(X_test)


# In[ ]:




