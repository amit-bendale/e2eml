#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[72]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


# In[3]:


df = pd.read_csv("./auto+mpg/auto-mpg.data",sep=" ",na_values='?',comment="\t",skipinitialspace=True, header=None)


# In[4]:


df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','year','origin']
df.head()


# In[80]:


class PrepTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        origin_encoder = OneHotEncoder(handle_unknown='ignore')
        origin_encoder.fit(X['origin'].values.reshape(-1, 1))
        self.origin_encoder = origin_encoder
        
        self.standardised_features = ['cylinders','displacement','horsepower','weight','acceleration','year']
        self.standard_scalers = [StandardScaler()]*len(self.standardised_features)
        for ind,feat in enumerate(self.standardised_features):
            self.standard_scalers[ind].fit(X[feat].values.reshape(-1,1))
        
        self.hp_imputer = SimpleImputer(strategy='median')
        self.hp_imputer.fit(X['horsepower'].values.reshape(-1,1))
        
    def transform(self, X):
        origin_ohe = self.origin_encoder.transform(X['origin'].values.reshape(-1,1)).toarray()
        X['acc_on_cyl'] = X['acceleration']/X['cylinders']
        X['acc_on_disp'] = X['acceleration']/X['displacement']
        X_cols_retain = [c for c in X.columns.tolist() if c!='origin']
        cols = (X_cols_retain
            + ['origin_{}'.format(x) for x in range(origin_ohe.shape[-1])])
        
        combined = np.column_stack((X[X_cols_retain],origin_ohe))
        temp_df = pd.DataFrame(combined,columns=cols) 
        
        # standard scaling
        for ind,feat in enumerate(self.standardised_features):
            temp_df[feat] = self.standard_scalers[ind].transform(X[feat].values.reshape(-1,1))
        
        temp_df['horsepower']=self.hp_imputer.transform(temp_df['horsepower'].values.reshape(-1,1))
        return temp_df


# In[81]:


prepTransformer = PrepTransformer()
prepTransformer.fit(df)


# In[82]:


df_1 = prepTransformer.transform(df)


# In[83]:


features = [c for c in df_1.columns if c!='mpg']
model = LinearRegression().fit(X=df_1[features],
                               y=df_1['mpg'])


# In[88]:


model.score(df_1[features],df_1['mpg'])


# In[90]:


import pickle
with open('model.bin','wb') as mout:
    pickle.dump(model, mout)


# In[91]:


with open('prep.bin','wb') as pout:
    pickle.dump(prepTransformer, pout)


# In[ ]:




