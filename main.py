import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


DATA_X = [1, 2, 3, 4]  # Enter your data for X here (args).
DATA_Y = [1, 2, 3, 6]  # Enter your data for X here (results).

CAPTION_X = ''  # Your X caption here.
CAPTION_Y = ''  # Your Y caption here.

# Data
data = {
    'X': DATA_X,
    'Y': DATA_Y,
}

df = pd.DataFrame(data)
print(df)  # X, Y Table output.

# X days and Y candies
X = df[['X']]
Y = df['Y']

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
plt.xlabel(CAPTION_X)
plt.ylabel(CAPTION_Y)
plt.show()

# Forecast
new_value = [[10]]
predicted_candies = model.predict(new_value)
print(f'Forecast for {new_value}:', predicted_candies[0])

