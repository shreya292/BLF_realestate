#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action='ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error

from sklearn.model_selection import GridSearchCV,KFold

from sklearn.feature_selection import SelectKBest, f_regression,RFE
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

import pickle


# In[3]:


df = pd.read_csv('House Data.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.duplicated().sum()


# In[9]:


df.isnull().sum().sum()


# In[10]:


df = df.drop(['Unnamed: 0'],axis=1)


# In[11]:


df = df.drop(['Id'],axis=1)


# In[12]:


df.head()


# In[13]:


sns.displot(df['SalePrice'])


# In[14]:


sns.distplot(np.log(df['SalePrice']))
plt.show()


# In[15]:


df['SaleCondition'].value_counts()


# In[16]:


plt.figure(figsize = [6, 3])
sns.barplot(x = df['SaleCondition'], y = df['SalePrice'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[17]:


plt.figure(figsize = [6, 3])
sns.barplot(x = df['Neighborhood'], y = df['SalePrice'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[18]:


plt.figure(figsize = [6, 3])
sns.barplot(x = df['BsmtQual'], y = df['SalePrice'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[19]:


plt.figure(figsize = [6, 3])
sns.barplot(x = df['BsmtCond'], y = df['SalePrice'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[20]:


plt.figure(figsize = [6, 3])
sns.barplot(x = df['ExterQual'], y = df['SalePrice'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[21]:


plt.figure(figsize = [6, 3])
sns.barplot(x = df['ExterCond'], y = df['SalePrice'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[22]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[23]:


df = df.drop(['SalePrice_log'],axis=1)


# In[24]:


df = df.drop(['GarageCars'],axis=1)


# In[25]:


df = df.drop(['1stFlrSF'],axis=1)


# In[26]:


house_num=df.select_dtypes(include=['float64','int64'])
house_num_columns=list(house_num)
print("No. of Numerical Features are: ",house_num.shape[1])
house_num_columns


# In[27]:


house_cat=df.select_dtypes(include=['object'])
house_cat_columns=list(house_cat)
print("No. of Categorical Features are: ",house_cat.shape[1])
house_cat_columns


# In[28]:


house_num_corr=house_num.corr()['SalePrice'][:-1]


# In[29]:


top_features=house_num_corr[abs(house_num_corr)>0.5].sort_values(ascending=False)


# In[30]:


print("There is {} strongly correlated values with salePrice:\n{}".format(len(top_features),top_features))


# In[31]:


for i in range(0,len(house_num.columns),5):
    sns.pairplot(data=house_num,x_vars=house_num.columns[i:i+5],y_vars=['SalePrice'])


# In[32]:


df['ExterQual'].value_counts()


# In[33]:


df['BsmtQual'].value_counts()


# In[34]:


df['OverallCond'].value_counts()


# In[35]:


df['ExterCond'].value_counts()


# In[36]:


df['OverallQual'].value_counts()


# In[37]:


qual_counts = df['OverallCond'].value_counts()
print(qual_counts)


# In[38]:


df['TotRmsAbvGrd'].value_counts()


# In[39]:


df['ExterQual'].replace(['Ex', 'Gd', 'TA', 'Fa'], [4, 3, 2, 1], inplace=True)
df['BsmtQual'].replace(['Ex', 'Gd', 'TA', 'Fa','No Basement'], [4, 3, 2, 1, 0], inplace=True)
df['ExterCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1], inplace=True)
df['OverallCond'].replace(['Excellent', 'Above Average', 'Very Good', 'Good','Average','Fair', 'Below Average','Poor', 'Very Poor'],
                              [9, 8, 7, 6, 5, 4, 3, 2, 1], inplace=True)
df['OverallQual'].replace(['Very Excellent','Excellent', 'Above Average', 'Very Good', 'Good','Average','Fair', 'Below Average','Poor', 'Very Poor'],
                              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], inplace=True)


# In[40]:


col= ['OverallQual','ExterQual','BsmtQual','LotArea','GrLivArea','GarageArea','TotalBsmtSF','TotRmsAbvGrd','YearBuilt','SalePrice']
df=df[col]
X = df.iloc[:, 0:9]
y = df.iloc[:,9:]


# In[41]:


X


# In[42]:


X.info()


# In[43]:


y


# In[44]:


y.info()


# In[45]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[46]:


models={}


# In[47]:


#Linear regression
LinReg= LinearRegression()

LinReg.fit(X_train,y_train)

y_pred = LinReg.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['LinearRegression'] = LinReg.score(X_test, y_test)


# In[48]:


#Lasso Regression

LasReg = Lasso(alpha=0.001)

LasReg.fit(X_train,y_train)

y_pred = LasReg.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['Lasso Regression'] = LasReg.score(X_test, y_test)


# In[49]:


#Ridge Regression
RidReg = Ridge(alpha=0.001)

RidReg.fit(X_train,y_train)

y_pred = RidReg.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['Ridge Regression'] = RidReg.score(X_test, y_test)


# In[50]:


#KNN
KNN = KNeighborsRegressor(n_neighbors=3)

KNN.fit(X_train,y_train)

y_pred = KNN.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['KNeighboursRegressor'] = KNN.score(X_test, y_test)


# In[51]:


#decision Tree
DT = DecisionTreeRegressor(max_depth=8)

DT.fit(X_train,y_train)

y_pred = DT.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['Decision Tree Regressor'] = DT.score(X_test, y_test)


# In[52]:


#SVM
SVM= SVR(kernel='rbf',C=100000,epsilon=0.01)

SVM.fit(X_train,y_train)

y_pred = SVM.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['Support Vector Machine'] = SVM.score(X_test, y_test)


# In[53]:


#Random Forest
RF= RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

RF.fit(X_train,y_train)

y_pred = RF.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['Random Forest Regressor'] = RF.score(X_test, y_test)


# In[54]:


#Gradient Boosting  Regressor
GBR = GradientBoostingRegressor(n_estimators=100,
                                  max_depth=15,
                                  min_samples_split=2,
                                  learning_rate=0.1,
                                  loss='absolute_error')

GBR.fit(X_train,y_train)

y_pred = GBR.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['Gradient BRegressor'] = GBR.score(X_test, y_test)


# In[55]:


#XGB Regressor
XGB = XGBRegressor(n_estimators=1000,learning_rate=0.1)

XGB.fit(X_train,y_train)

y_pred = XGB.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_percentage_error(y_test,y_pred))

models['Extreme GB Regressor'] = XGB.score(X_test, y_test)


# In[56]:


print("Model           \t      Accuracy")
for model, accuracy in models.items():
    print(f"{model}\t{accuracy}")


# In[57]:


with open ("module.pkl","wb") as f:
    pickle.dump(LinReg,f)

