# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SELVAGANESH R
RegisterNumber:  212223230200
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![Screenshot 2024-08-24 083303](https://github.com/user-attachments/assets/4b8a7976-cff7-4ec7-b45b-e7957a0e37b1)

![Screenshot 2024-08-24 083355](https://github.com/user-attachments/assets/d20502a0-f1dd-4faf-91d0-047115267df5)

![Screenshot 2024-08-24 083437](https://github.com/user-attachments/assets/f178b28c-cc28-42aa-9287-36cb78147b4f)

![Screenshot 2024-08-24 083522](https://github.com/user-attachments/assets/1d55df18-e47e-49ae-a605-c37baf27c4ad)

![Screenshot 2024-08-24 083552](https://github.com/user-attachments/assets/dffd7ebb-18d4-4e98-96f2-85a90ed93f0a)

![Screenshot 2024-08-24 083622](https://github.com/user-attachments/assets/139f1a70-b4be-4b03-87a7-513df1092cf6)

![Screenshot 2024-08-24 083655](https://github.com/user-attachments/assets/ea3310e9-ee9c-4224-82e0-d240cb1a7a67)

![Screenshot 2024-08-24 083724](https://github.com/user-attachments/assets/1c8a70ce-eec2-4f2d-b2a6-6ffa37372531)

![Screenshot 2024-08-24 083750](https://github.com/user-attachments/assets/d0e84d4c-f107-4e33-9e3f-24e0d9853c84)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
