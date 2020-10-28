#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


dataframe = pd.read_csv("Downloads\Mark2.csv")


# In[9]:


dataframe.head()


# In[10]:


dataframe.shape


# In[11]:


dataframe.describe()


# In[12]:


sns.pairplot(dataframe[['cough','chest_pain','fatigue','back_pain','acidity','prognosis']],hue='prognosis')


# In[13]:


train=dataframe.drop('prognosis',axis=1)
train.head()


# In[14]:


prognosis=dataframe.prognosis
prognosis.head()


# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(train,prognosis, test_size=0.2, random_state=0)
print("X_train size  ==>", X_train.shape)
print("X_test size  ==>", X_test.shape)
print("Y_train size  ==>", Y_train.shape)
print("Y_test size  ==>", Y_test.shape)


# In[16]:


clf1=svm.SVC(kernel='linear')
clf1.fit(X_train,Y_train)
Y_pred=clf1.predict(X_test)


# In[18]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred,average='micro'))


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)


# In[23]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred,average='micro'))


# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)


# In[26]:


Y_pred = classifier.predict(X_test)


# In[28]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred,average='micro'))


# In[29]:


import pickle


# In[30]:


with open('mark11_pickle','wb') as f:
    pickle.dump(clf1,f)


# In[31]:


with open('mark11_pickle','rb') as f:
    mp = pickle.load(f)


# In[32]:


mp.predict(X_test)


# In[ ]:




