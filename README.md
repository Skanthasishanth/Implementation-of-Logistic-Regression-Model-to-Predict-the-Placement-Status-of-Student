# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3. Import LabelEncoder and encode the dataset.

4. Import LogisticRegression from sklearn and apply the model on the dataset.

5. Predict the values of array.

6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7. Apply new unknown values.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S Kantha Sishanth
RegisterNumber: 212222100020
```
```py
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) # Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### Placement Data

![mlexp4_1](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/63b388c3-fea3-4285-aed1-085e0a7d203f)

### Salary Data

![mlexp4_2](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/7a6d2e4f-3682-4277-a6bc-1597dc005319)

### Checking the null function()

![mlexp4_3](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/5b8a3a79-9eb8-45a9-bf70-fa89cd791e7c)

### Data Duplicate

![mlexp4_4](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/ed624bb4-dea0-48c2-aec6-1d49e2e8731e)

### Print Data

![mlexp4_5](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/56e6fe4c-56ec-4f4e-8e7f-4f9f0ae12ada)

### Data Status

![mlexp4_6](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/a924580d-0958-450c-88b8-e6fd1c150820)

### y_prediction array

![mlexp4_7](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/4d7bfcdc-7a59-4cd3-8a75-f93a12630eea)

### Accuracy Value

![mlexp4_8](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/695f8d19-de52-4432-ae26-a04aa8167f55)

### Confusion Matrix

![mlexp4_9](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/1cc76c71-0b01-40fe-9bee-a2151fcbf9b1)

### Classification Report

![mlexp4_10](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/4df6fcad-ffa3-42e2-9be0-d35fd0c06d33)

### Prediction of LR

![mlexp4_11](https://github.com/Skanthasishanth/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118298456/5693d8ad-c5c0-4afc-b10c-ee2745ee63e7)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
