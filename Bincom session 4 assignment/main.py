import pandas as pd
from io import StringIO
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample CSV data as a multi-line string
sample_csv = '''
Fieldsite,Temperature,Salinity
Hobbs River,34,20
Logan River,20,30
cherry creek,33,27
Sacramento River,34,25
Mississippi River,29,35
Fields Creek,38,26
'''

# Use StringIO to read the CSV data into a DataFrame
data = StringIO(sample_csv)
df = pd.read_csv(data)

# Display the DataFrame
print(df)

# Extract features and target variable
X = df[['Temperature']].values  # Independent variable
y = df['Salinity'].values       # Dependent variable

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict salinity based on the temperature
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Temperature')
plt.ylabel('Salinity')
plt.title('Linear Regression: Temperature vs Salinity')
plt.legend()
plt.show()
