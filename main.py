import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Генеруємо дані
np.random.seed(24)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
energy_usage = 10 * hours + 15 + np.random.normal(0, 5, len(hours))  # Додаємо шум


# Вхідні та вихідні дані
X = hours.reshape(-1, 1)  # Перетворюємо в 2D-масив
Y = energy_usage  # Вихідні дані


# Creating and learning of Model
model = LinearRegression()
model.fit(X, Y)

# Output of coefficients 
print('W: ', model.coef_)
print('b', model.intercept_)

# Forecasting
Y_pred = model.predict(X)

# Graph with the line of regression
plt.scatter(X, Y, color='blue', label='Real data')
plt.plot(X, Y_pred, color='red')
plt.title('Line regression')
plt.xlabel('Години')
plt.ylabel('Витрати електроенергії')
plt.show()

# Forecast
new_value = [[10]]
predicted_candies = model.predict(new_value)
print(f'Forecast for {new_value}:', predicted_candies[0])

