#!/usr/bin/env python
# coding: utf-8

# # AP Final Project

# ## importing relevant libraries

# In[140]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats


# ### Uploading dataset

# In[141]:


df=pd.read_csv('laptops.csv',encoding='latin-1')


# In[142]:


df.head()


# ### Dropping the first column

# In[143]:


df.drop(df.columns[0],axis=1,inplace=True)


# In[144]:


df.sample(10)


# ### Checking statistics of the dataset

# In[145]:


df.shape


# In[146]:


df.isnull().sum()


# No missing values 

# In[147]:


df.describe(include='all')


# In[148]:


df.info()


# #### checking datatypes for attributes

# In[149]:


df.dtypes


# ## Data Cleaning

# #### cleaning memory column

# In[150]:


df['Memory']=df['Memory'].str.replace('1.0','1')
df['Memory']=df['Memory'].str.replace('Hybrid','HDD')


# #### converting weight to numeric

# In[151]:


df['Weight'].replace('kg','',True,regex=True)


# In[152]:


df.rename(columns={'Weight':'weight_kgs'},inplace=True)


# In[153]:


df.weight_kgs=df.weight_kgs.astype(float)


# #### euros to rupees conversion

# In[154]:


df['price_in_rs']=df['Price_euros']*84.75


# In[155]:


df.drop('Price_euros',axis=1,inplace=True)


# #### Converting to int

# In[156]:


df.Ram.replace('(GB)$','',regex=True,inplace=True)
df.Ram=df.Ram.astype(int)


# In[157]:


df.head()


# #### reducing memory down to 4 values

# In[158]:


df.Memory.value_counts()


# In[159]:


import re


# In[160]:


for i in range(len(df['Memory'])):
    #print(df.Memory[i])
    x=re.search('[/+]',df.Memory[i])
    if x!=None:
        df.Memory[i]=str('SSD and HDD')


# In[161]:


for i in range(len(df['Memory'])):
       df.Memory[i]=df.Memory[i].split(' ')[1] 


# In[162]:


df.Memory.replace('and','SSD and HDD',inplace=True)


# In[163]:


df.Memory.value_counts()


# In[164]:


df.head()


# In[165]:


df_cleaned=df.copy()


# In[166]:


df_cleaned.head()


# In[360]:


df_for_web=df_cleaned.copy()


# In[361]:


df_for_web.drop(['Product','Inches','ScreenResolution','Cpu','Gpu','weight_kgs'],axis=1,inplace=True)


# ## Exploratory Data Analysis

# ### checking correlations

# In[167]:


(df_cleaned[['Inches','weight_kgs','Ram','price_in_rs']]).corr()


# ##### correlation matrix

# In[170]:


sns.heatmap(df_cleaned[['Inches','weight_kgs','Ram','price_in_rs']].corr(),annot=True)
plt.show()


# ### Plotting attributes against each other

# #### weight vs price

# In[174]:


sns.regplot(data=df_cleaned,x='weight_kgs',y='price_in_rs')
plt.show()


# #### ram vs prices

# In[175]:


sns.regplot(data=df_cleaned,x='Ram',y='price_in_rs')
plt.show()


# #### inches vs price

# In[176]:


sns.regplot(data=df_cleaned,x='Inches',y='price_in_rs')
plt.show()


# ### Checking company wise prices across variants

# In[186]:


df_1=df[['Company','TypeName','price_in_rs']]
grouped=df_1.groupby(['Company','TypeName'],as_index=False).mean()
pivoted=grouped.pivot(index='Company',columns='TypeName').replace(np.nan,'0').astype(float)
pivoted


# ### Comparing companies on price range

# In[193]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Company',y='price_in_rs',data=df_cleaned)
plt.xticks(rotation=45)
plt.show()


# ###  types with price

# In[194]:


plt.figure(figsize=(10,6))
sns.boxplot(x='TypeName',y='price_in_rs',data=df_cleaned)
plt.xticks(rotation=45)
plt.show()


# ### Price across OS

# In[195]:


plt.figure(figsize=(10,6))
sns.boxplot(x='OpSys',y='price_in_rs',data=df_cleaned)
plt.xticks(rotation=45)
plt.show()


# ## Preparing data for model building

# In[236]:


df_prepare=df_cleaned.copy()


# In[237]:


df_prepare.head()


# In[238]:


df_prepare=pd.get_dummies(df_prepare,columns=['Company','OpSys','TypeName','Memory'],drop_first=True,prefix='',prefix_sep='')


# In[239]:


df_prepare.head()


# In[241]:


df_final=df_prepare.drop(['Product','Inches','ScreenResolution','Cpu','Gpu','weight_kgs'],axis=1)


# In[242]:


df_final.head()


# ## Model Building

# In[316]:


x=df_final.drop(['price_in_rs'],axis=1)
y=df_final['price_in_rs']


# In[317]:


x.head()


# In[318]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# ### Finding best model using GridSearchCV:

# In[353]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators':[100,250,500,1000],
                 'max_depth':[None,10,15,50],
                'max_features':[0.75],
                'max_samples':[0.5],
                
              
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = KFold(n_splits=5,shuffle=True)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)


# In[322]:


x_test.shape,y_test.shape


# In[357]:


rf=RandomForestRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=50)
rf.fit(x,y)
rf.score(x_test,y_test)


# ## Predictions

# In[431]:


def predict_price(Ram,Company,TypeName,Memory,OpSys):    
    loc_index = np.where(x.columns==Company)[0][0]
    loc_index1 = np.where(x.columns==TypeName)[0][0]
    loc_index2 = np.where(x.columns==Memory)[0][0]
    loc_index3 = np.where(x.columns==OpSys)[0][0]
    
    X = np.zeros(len(x.columns))
    X[0] = Ram
    #X[1]=Memory

    if loc_index >= 0:
        X[loc_index] = 1
    if loc_index1 >= 0:
        X[loc_index1] = 1
    if loc_index2 >= 0:
        X[loc_index2] = 1
    if loc_index3 >= 0:
        X[loc_index3] = 1
    
    print('\nPrice: ',rf.predict([X])[0]) 


# In[432]:


predict_price(input('Enter RAM: '),input('Enter Brand: '),input('Enter Type: '),input('Enter Memory type: '),input('Enter OS: '))


# In[403]:


with open('columns.pickle','wb') as f:
    pickle.dump(df_for_web,f)


# In[429]:


import pickle
with open('laptop_price_prediction_model.pickle','wb') as f:
    pickle.dump(pipe,f)

