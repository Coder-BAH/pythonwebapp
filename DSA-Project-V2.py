#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[138]:


df=pd.read_csv('movie_metadata.csv')
df.head()


# In[139]:


df.info()


# In[140]:


#Summary for the numerical columns in the dataset
df.describe().T


# In[141]:


df.isnull().sum()


# In[142]:


df.drop_duplicates(inplace = True)
df.shape


# In[143]:


# We can remove the null values from the dataset where the count is less . so that we don't loose much data 
df.dropna(axis=0,subset=['director_name', 'num_critic_for_reviews','duration','director_facebook_likes','actor_2_name','actor_1_name','actor_3_name','facenumber_in_poster','num_user_for_reviews','language','country','plot_keywords'],inplace=True)


# In[144]:


df.shape


# In[145]:


df.info()


# #Handling Missing values in the dataset
# 

# In[146]:


#Replacing the content rating with Value R as it has highest frequency

df["color"].fillna(df["color"].mode().iloc[0], inplace = True) 


# In[147]:


#Replacing the content rating with Value R as it has highest frequency

df["content_rating"].fillna(df["content_rating"].mode().iloc[0], inplace = True) 


# In[148]:


#Replacing the aspect_ratio with the median of the value as the graph is right skewed 

df["aspect_ratio"].fillna(df["aspect_ratio"].median(),inplace=True)


# In[149]:


#We need to replace the value in budget with the median of the value
df["budget"].fillna(df["budget"].median(),inplace=True)


# In[150]:


# We need to replace the value in gross with the median of the value 

df['gross'].fillna(df['gross'].median(),inplace=True)


# In[151]:


# We can remove the null values from the dataset where the count is less . so that we don't loose much data 
df.dropna(axis=0,subset=['director_name', 'num_critic_for_reviews','duration','director_facebook_likes','actor_2_name','actor_1_name','actor_3_name','facenumber_in_poster','num_user_for_reviews','language','country','plot_keywords'],inplace=True)


# In[152]:


# Recheck that all the null values are removed

df.isna().sum()


# In[153]:


df.shape


# Categorical and numerical columns in the dataset

# In[154]:


numerical_cols = [col for col in df.columns if df[col].dtype != 'object']
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']


# In[155]:


categorical_cols, numerical_cols


# In[156]:


#Plotting the Correlation between the numerical values of the Dataset
_, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), linewidths=.6, cmap='coolwarm', annot=True, ax=ax)


# In[157]:


#Making a new column named cast$crew total facebook likes by adding columns director_facebook_likes,actor_3_facebook_likes,actor_2_facebook_likes,cast_total_facebook_likes
df['cast$crew_total_facebook_likes']=df['director_facebook_likes']+df['actor_3_facebook_likes']+df['actor_2_facebook_likes']+df['cast_total_facebook_likes']
df.head(2) 


# In[158]:


df.drop(columns=['director_facebook_likes','actor_3_facebook_likes','cast_total_facebook_likes','actor_2_facebook_likes'],axis=1,inplace=True)
df.head(2)
    


# In[ ]:





# In[23]:


df['director_name'].value_counts()


# In[159]:


#Removing the director name column

df.drop('director_name', axis=1, inplace=True)


# In[160]:


#Removing the actor1 ,actor 2 and actor 3 names 

df.drop('actor_1_name',axis=1,inplace=True)


# In[161]:


df.drop('actor_2_name',axis=1,inplace=True)


# In[162]:


df.drop('actor_3_name',axis=1,inplace=True)


# In[163]:


#Dropping the movie title 

df.drop('movie_title',axis=1,inplace=True)


# In[164]:


# Dropping the plot keywords
df.drop('plot_keywords',axis=1,inplace=True)


# In[165]:


#Dropping the imdb link as is it is unique value
df.drop('movie_imdb_link',axis=1,inplace=True)


# In[166]:


#Dropping column country as most of the movies are in color
df.drop('country',axis=1,inplace=True)


# In[167]:


#Dropping column aspect ratio
df.drop('aspect_ratio',axis=1,inplace=True)


# In[168]:


#Dropping column color as most of the movies are in color
df.drop('color',axis=1,inplace=True)


# In[169]:


df.drop('title_year',axis=1,inplace=True)


# In[170]:



#Most of the values for the languages is english we can drop the english column

df.drop('language',axis=1,inplace=True)


# We want to keep num_voted_users and take the ratio of num_user_for_reviews and num_critic_for_reviews.

# In[171]:


#Ratio of the ratio of num_user_for_reviews and num_critic_for_reviews.

df['critic_review_ratio']=df['num_critic_for_reviews']/df['num_user_for_reviews']


# In[172]:


#Dropping the num_critic_for_review

df.drop('num_critic_for_reviews',axis=1,inplace=True)
df.drop('num_user_for_reviews',axis=1,inplace=True)


# In[173]:


df.shape


# In[174]:


numerical_cols = [col for col in df.columns if df[col].dtype != 'object']
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']


# In[175]:


categorical_cols, numerical_cols


# In[176]:


#Plotting the Correlation between the numerical values of the Dataset
_, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), linewidths=.6, cmap='coolwarm', annot=True, ax=ax)


# In[177]:


df.head()


# In[178]:


df2=df.copy()
df2.head(2)


# In[102]:


#Label encoding the categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_list=['genres','content_rating']
df[cat_list]=df[cat_list].apply(lambda x:le.fit_transform(x))


# In[103]:


#A sample of data after label encoding
df.head()


# In[104]:


#Plotting the Correlation between the numerical values of the Dataset
_, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), linewidths=.6, cmap='coolwarm', annot=True, ax=ax)


# In[105]:


df1=df.copy()


# In[106]:


# split the dataset into dependent(X) and Independent(Y) datasets
X=df1.drop(['imdb_score'],axis=1)
y = df1['imdb_score']


# In[107]:


# spliting the data into trainning and test dateset
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[108]:


# feature scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# In[109]:


#Linear Regressor
from sklearn import linear_model
lr=linear_model.LinearRegression()
model=lr.fit(X_train,y_train)
predictions=model.predict(X_test)


# In[110]:


from sklearn.metrics import mean_squared_error
print('MSE is:',mean_squared_error(y_test,predictions))


# In[111]:


from sklearn.metrics import r2_score as r2
print(r2(y_test,predictions))


# In[112]:


#Visualization of test and prediction data
plt.figure(figsize=(5,5))
plt.scatter(y_test,predictions)


# In[113]:


from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 500)
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)


# In[114]:


mean_squared_error(rf_pred, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[179]:


df2=df.copy()


# In[180]:


df2.head()


# Transforming the right skwed features into logarithmic scale

# In[181]:


df2['budget'] = np.log(df2['budget'])


# In[182]:


df2['gross'] = np.log(df2['gross'])


# In[183]:


df2['num_voted_users'] = np.log(df2['num_voted_users'])


# In[184]:


df2.genres.unique(), df2.genres.nunique()


# In[185]:


df2['main_genre'] = df2['genres'].str.split('|').str[0]


# In[186]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2['main_genre'] = le.fit_transform(df2.main_genre)


# In[190]:


df2.drop('genres',axis=1,inplace=True)


# In[191]:


df2.head(2)


# In[192]:


le1 = LabelEncoder()
df2['content_rating'] = le1.fit_transform(df2.content_rating)


# In[193]:


df2.head(2)


# In[194]:


df2.columns


# In[195]:


# split the dataset into dependent(X) and Independent(Y) datasets
X=df2.drop(['imdb_score'],axis=1)
y = df2['imdb_score']


# In[196]:


# spliting the data into trainning and test dateset
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[197]:


#Linear Regressor
from sklearn import linear_model
lr=linear_model.LinearRegression()
model=lr.fit(X_train,y_train)
predictions=model.predict(X_test)


# In[198]:


from sklearn.metrics import mean_squared_error
print('MSE is:',mean_squared_error(y_test,predictions))


# In[199]:


from sklearn.metrics import r2_score as r2
print(r2(y_test,predictions))


# In[200]:


#Visualization of test and prediction data
plt.figure(figsize=(5,5))
plt.scatter(y_test,predictions)


# In[201]:


from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 500)
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)


# In[202]:


mean_squared_error(rf_pred, y_test)


# Hyperparameter tuning with random forest regressor

# In[210]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [90, 100],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 500, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search_rf.fit(X_train, y_train)


# In[211]:


grid_search_rf.fit(X_train, y_train)
grid_search_rf.best_params_


# In[212]:


y_grid_pred_rf = grid_search_rf.predict(X_test)
mean_squared_error(y_grid_pred_rf, y_test.values)


# # XGBoost with Hyperparameter tuning

# In[214]:


get_ipython().system('pip3 install xgboost')


# In[215]:


import xgboost as xgb
xg_model = xgb.XGBRegressor(n_estimators = 500)
xg_model.fit(X_train, y_train)


# In[216]:


results = xg_model.predict(X_test)


# In[218]:


xg_model.score(X_train, y_train)


# In[224]:


from sklearn.metrics import r2_score
r2_score(y_test, results)


# In[225]:


from sklearn.metrics import mean_squared_error
print('MSE is:',mean_squared_error(y_test,results))


# In[220]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [3, 4],
    'learning_rate' : [0.1, 0.01, 0.05],
    'n_estimators' : [100, 500, 1000]
}
# Create a based model
model_xgb= xgb.XGBRegressor()
# Instantiate the grid search model
grid_search_xgb = GridSearchCV(estimator = model_xgb, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[221]:


grid_search_xgb.fit(X_train, y_train)
grid_search_xgb.best_params_


# In[222]:


y_pred_xgb = grid_search_xgb.predict(X_test)


# In[223]:


mean_squared_error(y_test.values, y_pred_xgb)


# In[227]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred_xgb)


# In[226]:


#Visualization of test and prediction data
plt.figure(figsize=(5,5))
plt.scatter(y_test,y_pred_xgb)


# In[ ]:





# In[ ]:





# In[ ]:





# # Gradient Boosting Regressor with hyper parameter tuning

# In[203]:


from sklearn import ensemble
n_trees=200
gradientboost = ensemble.GradientBoostingRegressor(loss='ls',learning_rate=0.03,n_estimators=n_trees,max_depth=4)
gradientboost.fit(X_train,y_train)


# In[204]:


y_pred_gb=gradientboost.predict(X_test)
error=gradientboost.loss_(y_test,y_pred_gb) ##Loss function== Mean square error
print("MSE:%.3f" % error)


# Hyperparameter tuning of gradient boosting regressor

# In[205]:


y_pred_gb.min(), y_pred_gb.max()


# In[206]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'loss' : ['ls'],
    'max_depth' : [3, 4, 5],
    'learning_rate' : [0.01, 0.001],
    'n_estimators': [100, 200, 500]
}
# Create a based model
gb = ensemble.GradientBoostingRegressor()
# Instantiate the grid search model
grid_search_gb = GridSearchCV(estimator = gb, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[207]:


grid_search_gb.fit(X_train, y_train)
grid_search_gb.best_params_


# In[208]:


grid_search_gb_pred = grid_search_gb.predict(X_test)


# In[209]:


mean_squared_error(y_test.values, grid_search_gb_pred)


# In[ ]:





# # Conclusion

# Linear Regressor:           MSE- 0.79 ,  r2 - 0.37
# Random Forest Regressor:    MSE- 0.61 ,  r2 -
# XGBoost Regressor:          MSE- 0.65 ,  r2 -0.48
# Gradient Boost Regressor:   MSE- 0.60 ,  r2 -

# After Hyper parameter tuning
# Random Forest Regressor:    MSE- 0.60 ,  
# XGBoost Regressor:          MSE- 0.57 ,  r2=.54
# Gradient Boost Regressor:   MSE- 0.60 ,

# So after hyper parameter tuning XGBoost regressor shows less error . So we can consider XGBoost as the final model

# In[ ]:





# # Interpreting Results of Regression Model
# Considering XG Boost as a final model with very less error rate.

# In[229]:


feature_importance = grid_search_xgb.best_estimator_.feature_importances_
sorted_importance = np.argsort(feature_importance)
pos = np.arange(len(sorted_importance))
plt.figure(figsize=(12,5))
plt.barh(pos, feature_importance[sorted_importance],align='center')
plt.yticks(pos, X_train.columns[sorted_importance],fontsize=15)
plt.title('Feature Importance ',fontsize=18)
plt.show()


# In[ ]:





# In[ ]:





# # Exporting the data

# model_xgb= xgb.XGBRegressor()
# model.fit((X_train, y_train))
# grid_search_xgb = GridSearchCV(estimator = model_xgb, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)
# grid_search_xgb.fit(X_train, y_train)
# y_pred_xgb = grid_search_xgb.predict(X_test)
# r2_score(y_test, y_pred_xgb)
# 

# In[236]:


xgb= xgb.XGBRegressor(n_estimators = 500)
xgb.fit(X_train, y_train)
results = xgb.predict(X_test)
r2_score(y_test, results)
print('MSE is:',mean_squared_error(y_test,results))
print('Score is:',r2_score(y_test, results))


# In[238]:


#Exporting
import pickle
pickle.dump(xgb,open('model.pkl','wb'))







