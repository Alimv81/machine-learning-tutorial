import csv
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Set the number of rows and parameters
n_rows = 100
n_params = 5

# Generate the parameter values and labels
params = np.random.rand(n_params)
labels = ['Param' + str(i+1) for i in range(n_params)]

# Generate the feature matrix X
X = np.round(np.random.rand(n_rows, n_params) * 100)

# Generate the target variable y
noise = np.random.normal(loc=0, scale=0.1, size=n_rows)
y = np.dot(X, params) + noise

# Print the first five rows of the data
print('X:')
print(X[:5, :])
print('\ny:')
print(y[:5])
print('\nLabels:')
print(labels)

with open('learning/lnrg1/data.csv', mode='w', newline=None) as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow([*labels, 'y'])
    writer.writerows(np.concatenate((X,y.reshape(-1, 100).T), axis=1))