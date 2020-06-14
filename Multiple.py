#Import Libraries
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

#Loading Dataset
dataset = pd.read_csv("Home_price.csv")
print(dataset) 

#Calculate the median 
#To covert integers into fload
a =math.floor( dataset.bedrooms.median())

#filling the na columns with median value
dataset.fillna(a ,inplace = True)
print(dataset)

#Creating Linear Regression Model
reg = LinearRegression()

#fit method to trainnig the model
reg.fit(dataset[['area','bedrooms','age']],dataset.price)

#prediciting the value
d=reg.predict([[3000,3,15]])
print(d)
#loading the prediciton Csv file
df = pd.read_csv("calculation.csv")
print(df)


#predicting the values in prediction csv
s = reg.predict(df)
print(s)

#new column
df['prices'] = s


#exporting the predicted value
df.to_csv("calculation.csv",index = False)
