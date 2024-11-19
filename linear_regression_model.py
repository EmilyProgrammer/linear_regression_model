import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib

#Importing dataset
df = pd.read_csv('data/FuelConsumption.csv')

#Data Preprocessing

#new_df = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'CO2EMISSIONS']]
#Independent and dependent features
X = df.iloc[:, [4,5,10]].values
y = df.iloc[:, -1].values
print(X)

print(y)

#Creating the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 2, test_size= 0.2)

#Building and Training the model
model = LinearRegression()

#Training the model
model.fit(X_train, y_train)

#Inference
y_pred = model.predict(X_test)

#Making prediction for a single data point ENGINESIZE = 3.9, CYLINDERS = 6, FUELCONSUMPTION_COMB = 12.2

model.predict([[3.9, 6, 12.2]])

#Evaluating the model
r2 = r2_score(y_test, y_pred)
print(r2)

#Adjusted R_squared
k = X_test.shape[1]
n = X_test.shape[0]

adj_r2 = 1 -(1 - r2) * (n-1)/(n - k- 1)
print(adj_r2)

#Scatter plot of Actual vs. Prediction values
plt.figure(figsize = (8,6)) #plot actual vs. predicted
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color ='red', linewidth = 2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted values')
plt.title('Actual vs. Predicted Values')
#plt.show()

#save the trained model to a .pkl (.pkl was not working, so I used .jlib)
joblib.dump(model, 'model/model.jlib')

