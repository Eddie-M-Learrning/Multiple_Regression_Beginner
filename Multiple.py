#Import Libraries
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Loading Dataset
dataset = pd.read_csv("Home_price.csv")
print(dataset)


 

#Calculate the median 
#To covert integers into fload
a =dataset.median()


#filling the na columns with median value
dataset.fillna(a.round(),inplace = True)
print(dataset)

x =dataset.iloc[:,:-1].values
y =dataset.iloc[:,-1].values
x_train ,x_test , y_train , y_test = train_test_split(x , y , test_size =0.33,random_state = 0)
#Creating Linear Regression Model
reg = LinearRegression()
reg.fit(x,y)
#prediciting the value
y_pred = reg.predict(x_test)
print(y_pred)



#loading the prediciton Csv file
df = pd.read_csv("calculation.csv")
print(df)

#predicting the values in prediction csv
#s = reg.predict(df)
#print(s.round())

#new column
#df['price'] = s.round()


#exporting the predicted value
#df.to_csv("calculation.csv",index = False)
am =reg.score(x_test,y_test)
print(am)

