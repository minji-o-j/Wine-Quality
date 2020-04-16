#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


get_ipython().system('pip install xgboost')


# In[7]:


get_ipython().system('pip install --upgrade pip')


# In[2]:


# Data manipulation
import numpy as np
import pandas as pd

# Data pre-processing
from sklearn.preprocessing import StandardScaler as ss

# Dimensionality reduction
from sklearn.decomposition import PCA

#  Data splitting and model parameter search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Modeling modules
from xgboost.sklearn import XGBClassifier

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance


# # 데이터 적재하기

# In[34]:


data=pd.read_csv("./data/1_winequalityN.csv")
dataw=pd.read_csv("./data/1_white.csv")
datar=pd.read_csv("./data/1_red.csv")

data=data.dropna(axis=0)  ##없애버림, 평균값 넣으려면 axis=0대신에 inplate=True 사용
dataw=dataw.dropna(axis=0)

print "wine 데이터셋의 크기:" ,data.shape
print "\nwine 데이터셋의 키:", data.keys()


# #### 데이터셋의 키
# ['type', 'fixed acidity', 'volatile acidity', 'citric acid',
#        'residual sugar', 'chlorides', 'free sulfur dioxide',
#        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
#        'quality']

# In[43]:


print "타깃(품질)의 크기:",data['quality'].shape
#print "타깃(품질):",data['quality']


# In[44]:


data.shape


# In[45]:


data.head() ##앞쪽 데이터 5개 보여줌


# In[47]:


data.tail() ##끝쪽 데이터 5개 <삭제되었어도 인덱스는 변하지 않는듯?


# In[48]:


data.info()


# # 데이터 살펴보기

# In[41]:


data.type.value_counts()


# In[40]:


data.quality.value_counts()
##quality에 대한 data가 3~9까지 있음을 확인할 수 있다.


# In[36]:


plt.figure(figsize=(12, 12))
sns.countplot(x = 'quality', data=data, hue='type')


# white에 비해 red는 품질 데이터 양이 매우 적으며 white도 3등급 9등급의 데이터 양이 매우 적다.

# In[37]:


#red+white
sns.pairplot(data, vars = ['fixed acidity', 'volatile acidity', 'citric acid','residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide','density'], hue='quality', size = 5, palette="prism")


# 대부분 점이 겹겹이 쌓여있어서 붓꽃 데이터처럼 명확하게 구분하는 것이 어려워보인다.
# 이산화황 관련된 특징들(free sulfur dioxide, total sulfur dioxide)이 데이터가 많이 퍼져있고, 특히 total sulfur dioxide의 경우 등급의 범위가 가장 광범위하게 퍼져있음

# In[38]:


#white일때만
sns.pairplot(dataw, vars = ['fixed acidity', 'volatile acidity', 'citric acid','residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide','density'], hue='quality', size = 5, palette="prism")


# # 훈련 데이터와 테스트 데이터
# ## 일단 전체 데이터로 했음

# In[157]:


#x_train에 적재하기 위한 2차원 배열의 특성 dataset을 만든다.
#6463*11(?) 사이즈

#실험의 편의성을 위해 data마다 이름을 붙였다.
f1=data['fixed acidity']
f2=data['volatile acidity']
f3=data['citric acid']
f4=data['residual sugar']
f5=data['chlorides']
f6=data['free sulfur dioxide']
f7=data['total sulfur dioxide']
f8=data['density']
f9=data['pH']
f10=data['sulphates']
f11=data['alcohol']


c_data=np.c_[f1,f2,f4,f8,f9,f10,f11]


# In[158]:


X_train, X_test, y_train, y_test = train_test_split(c_data, data['quality'],random_state=0)

print "X_train 크기: ", X_train.shape #(#(numbe) of data*75%, # of features)
print "y_train 크기: ", y_train.shape #x트레인 데이터에 대한 정답
print "X_test 크기: ", X_test.shape
print "y_test 크기: ", y_test.shape


# # k 최근접 이웃 알고리즘

# In[159]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train) ##train or fit, 학습시키기


# # 평가하기

# In[160]:


y_pred=knn.predict(X_test)
print "테스트 세트에 대한 예측값: \n", y_pred

print"테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred==y_test))


# ### feature에 따른 분석  
# 
# |<center>index</center>|<center>정확도</center>|
# |:---|:---:|
# |f1~11|0.56|
# |f1,f2|0.43|
# |f1~f3| 0.53|  
# |f1~f4| 0.54 | 
# |f1~f5| 0.53  |
# |f1~f6| 0.54  |
# |f1~f4,f6| 0.54|  
# |f1~f7| 0.54  |
# |f1~f8| 0.54  |
# |f1~f9| 0.54  |
# |f1~f10| 0.54  |
# |f1~f11| 0.56|
# |f3제외| 0.57 | 
# |f5제외| 0.56  |
# |f6,7제외| 0.60 | 
# |f3,5,6,7제외|<span style="color:red">0.61</span>|
# 
# - 하다보면 더 좋은 결과가 있을 것으로 예상함

# In[ ]:





# In[ ]:





# In[ ]:




