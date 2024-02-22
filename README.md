# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Assign the points for representing in the graph

5. Predict the regression for marks by using the representation of the graph.

6. Compare the graphs and hence we obtained the linear regression for the given data

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: BALAMURUGAN B
RegisterNumber: 212222230016

#IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.iloc[3])
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

#SPLITTING DATASET INTO TRAINING AND TESTING DATA
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#GRAPH PLOT FOR TRAINING DATA
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#GRAPH PLOT FOR TESTING DATA
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='black')
plt.title('Hours vs Scores(Testing set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

#CALCULATE MEAN SQUARED ERROR
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
#CALCULATE MEAN ABSOLUTE ERROR
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
#CALCULATE ROOT MEAN SQUARED ERROR
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
### 1.Dataset
![o1](https://github.com/BALA291/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120717501/94a2a5f3-c044-494f-8380-bb40172f3475)

### 2.Head
![head](https://github.com/BALA291/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120717501/e4d0146f-da6c-482a-912d-2fa4a28c9f75)

### 3.Tail
![tail](https://github.com/BALA291/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120717501/f89a2e9e-a82b-4937-9b4c-fee11dbda433)

### 4.X and Y values
![x and y](https://github.com/BALA291/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120717501/b7f70089-174d-4798-b1e6-a4f35c206e87)

### 5.Predicted values
![predicted](https://github.com/BALA291/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120717501/aad7bf35-798d-4614-b26f-1b856579dd56)

### 6. Error calculation
![errors](https://github.com/BALA291/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120717501/f450f215-771e-4069-b429-73b79a8119cc)

### 7.Training set
![training](https://github.com/BALA291/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120717501/668cb9c9-dac3-4e6c-8b34-b5f15e8be277)

### 8. Testing set
![testing](https://github.com/BALA291/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120717501/6a941bf6-5397-430b-8a70-1e8d7d3a6bc6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
