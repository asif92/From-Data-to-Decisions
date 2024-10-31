# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ice_Sales_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Tempratue vs Sales (Training set)')
plt.xlabel('Temprature')
plt.ylabel('Sales')
# plt.show() // uncomment if you are using Google Colab
plt.savefig('output/ice_cream_sales_train_set_plot_v2.png')  # Saves the plot as an image file


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Tempratue vs Sales (Test set)')
plt.xlabel('Temprature')
plt.ylabel('Sales')
# plt.show() // uncomment if you are using Google Colab
plt.savefig('output/ice_cream_sales_test_set_plot_v2.png')  # Saves the plot as an image file


# temprature = input("Enter temprature: ")
# print(regressor.predict([[temprature]]))
