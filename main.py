import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
data = {
    'Days': [1, 2, 3, 4, 5],
    'Candies': [2, 4, 6, 8, 10],
}

df = pd.DataFrame(data)
print(df)

# Visualisation of data
plt.scatter(df['Days'], df['Candies'], color='blue')
plt.title('Dependency graph')
plt.xlabel('Days')
plt.ylabel('Candies')
# plt.show()

# X days and Y candies
X = df[['Days']]
Y = df['Candies']

# Creating and learning of Model
model = LinearRegression()
model.fit(X, Y)

# Output of coefficients 
print('W: ', model.coef_)
print('b', model.intercept_)

# Forecasting
Y_pred = model.predict(X)

# Graph with the line of regression
plt.scatter(X, Y, color='blue')
plt.plot(X, Y_pred, color='red')
plt.title('Line regression')
plt.xlabel('Days')
plt.ylabel('Candies')
plt.show()

# Forecast
new_value = [[10]]
predicted_candies = model.predict(new_value)
print(f'Forecast for {new_value}:', predicted_candies[0])

