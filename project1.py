
"""
Created on Sun Jul 08 18:06:07 2018

@author: Jigyasa Yadav
"""
   
 
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading json data in dataframe
df = pd.read_json("demo.json",lines=True)
df1 = pd.read_json("test.json",lines=True)

#dtasets
ivtrain = df.iloc[:,1:].values
dvtrain = df.iloc[:,0].values.reshape(-1,1)
ivtest = df1.iloc[:,:].values

#viewing categories in dataframe
dvtrain1 = pd.DataFrame(dvtrain)
dvtrain1 = list(dvtrain1[0])

#labelencoding of feature
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lab=LabelEncoder()
for i in [0,1,2]:
    ivtrain[:,i]=lab.fit_transform(ivtrain[:,i])  

#one-hot encoding
hen=OneHotEncoder(categorical_features=[0,2])
ivtrain=hen.fit_transform(ivtrain).toarray()    


#label encoding of label data    
dvtrain=lab.fit_transform(dvtrain)

#zipping category and it's label encoded data
dvtrain2 = pd.DataFrame(dvtrain)
dvtrain2 = list(dvtrain2[0])
dictionary = dict(zip(dvtrain1,dvtrain2))



#label encoding of test data
for i in [0,1,2]:
    ivtest[:,i]=lab.fit_transform(ivtest[:,i])
    
#one hot encoding     
hen=OneHotEncoder(categorical_features=[0,2])
ivtest=hen.fit_transform(ivtest).toarray()  


#viewing features data
view = pd.DataFrame(ivtrain)
view1 = pd.DataFrame(ivtest)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
ivtrain = sc.fit_transform(ivtrain)
ivtest = sc.fit_transform(ivtest)


#random forest
from sklearn.ensemble import RandomForestRegressor
classi = RandomForestRegressor(n_estimators = 40,random_state = 0)
classi.fit( ivtrain,dvtrain)

#predictng values
pred = classi.predict(ivtrain)
pred1 = classi.predict(ivtest)

#checking accuracy
score = classi.score(ivtrain,dvtrain)

#predicting a random result
x = np.array(["newyork","map","for-sale"]).reshape(-1,1)
x = lab.fit_transform(x).reshape(1,-1)
x = hen.transform(x)
#x = sc.transform(x)
y = classi.predict(x)
z = int(round(y,0))
for key, value in dictionary.items():
  if z == value:
    print ("category of random result where we have provided city,heading and section: "+key)

#backward elimination
import statsmodels.formula.api as sm

features_opt = ivtrain[:,:]
reg_OLS = sm.OLS(endog = dvtrain, exog =features_opt ).fit()
reg_OLS.summary()


features_opt = ivtrain[:,[17,18,19,20]]
reg_OLS = sm.OLS(endog = dvtrain, exog =features_opt ).fit()
reg_OLS.summary()

print("printing coefficients :" , reg_OLS.params) #for printing coeff

#printing top 5 categories
labels = df['category'].value_counts()[:5]
l = list(labels.index)
v = list(labels.values)
print("5 top categories according to no. of posts: ",l)

#bar graph showing top 5 categories
colors = ['r','b','g','c','m']
plt.bar(l,v,color=colors)
plt.title('Bar graph showing 5 top most categories')
plt.xlabel('categories')
plt.ylabel('no. of posts')
plt.legend()
plt.show()
