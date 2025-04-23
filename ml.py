# My first Ml Project using sckit learn one of the powerful liberary and free
#  We are using dataset of delaney_solubility
# this dataset is crucial for the biologists and chemist

#First loading the datasets in our programme

import pandas as pd
df = pd.read_csv("C:\\Users\\Ashutosh Pandey\\Desktop\\solubility.csv")
# print(df)
#Data Preparation
# Data separation

y=df['logS']

# print(y)
#we have to drop the colum y in order to get x
x= df.drop('logS', axis=1) # axis=1 represents column and axis=0 represents rows

# print(x)

#Data spliting into training dataset and testing dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=100)

# print(x_train) // have 80percent of the data
# print(x_test)// have 20perecnt of data

#Model Building 
#linear Regression

#building and training the model;

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train, y_train) # its defines that we have to train our model on following dataset

# Applying the model to make a prediction

y_lr_train_pred = lr.predict(x_train) # making prediction on datset on which model is trained
y_lr_test_pred= lr.predict(x_test)

# print(y_lr_train_pred) # 80% of data is predicted
# print(y_lr_test_pred) # 20% of data is predicted


#Lets predict the original value with predictive value
# model performance
# print(y_train) #original
# print(y_lr_train_pred) #predicted

from sklearn.metrics import mean_squared_error, r2_score

#for training set
lr_train_mse=mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2= r2_score(y_train, y_lr_train_pred)

#for testing test

lr_test_mse= mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2= r2_score(y_test, y_lr_test_pred)

# print(lr_train_mse)
# print(lr_train_r2)
# print(lr_test_mse)
# print(lr_test_r2)

#lets make the resluts into tabular form

lr_results= pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
lr_results.columns = ['Method','Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
#the above line helps in visualizing it properly

# print(lr_results)

# Random Forest
#trainig the model
from sklearn.ensemble import RandomForestRegressor # since we are using the regressi

RF=RandomForestRegressor(max_depth=2, random_state=100)
RF.fit(x_train, y_train) # train the model

#applying the model to me a prediction 
y_RF_train_pred=RF.predict(x_train)
y_RF_test_pred=RF.predict(x_test)

#Evaluation model performance

from sklearn.metrics import mean_squared_error, r2_score

#for training
RF_train_mse = mean_squared_error(y_train, y_RF_train_pred)
RF_train_r2 = r2_score(y_train, y_RF_train_pred)

#for testing 
RF_test_mse = mean_squared_error(y_test, y_RF_test_pred)
RF_test_r2 = r2_score(y_test, y_RF_test_pred)

# making table for rest
RF_results= pd.DataFrame(['Rainforest Regression', RF_train_mse, RF_train_r2, RF_test_mse, RF_test_r2]).transpose()
RF_results.columns = ['Method','Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']

# print(RF_results)

#lets combine with both the video

df_models = pd.concat([lr_results, RF_results], axis=0).reset_index(drop=True)
print(df_models)

#Data visualizations of prediction results
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))

plt.scatter(x=y_train, y=y_lr_train_pred,c='#7CAE00', alpha=0.3)

z= np.polyfit(y_train, y_lr_train_pred, 1)
p= np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
plt.show()