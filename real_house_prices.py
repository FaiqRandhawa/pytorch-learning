#usin actual dataset from scikit learn, we will use california housing dataset to predict house prices. we will use a simple neural network with one hidden layer and train it using stochastic gradient descent optimizer and mean squared error loss function. we will also use pandas for data manipulation and analysis, and sklearn for data preprocessing and splitting the dataset into training and testing sets.

import torch
import numpy as np
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(42)  # Set random seed for reproducibility.
data = fetch_california_housing()  # Load the California housing dataset.

X=torch.FloatTensor(data.data)
y=torch.FloatTensor(data.target).unsqueeze(1)  # Convert target to a column vector.


scaler=StandardScaler()  # Create a StandardScaler object for feature scaling.
X_scaled=scaler.fit_transform(data.data)  # Fit the scaler to the data and transform it.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#converting to tensors now
X=torch.FloatTensor(X_scaled)  # Convert the scaled features to a PyTorch FloatTensor.
y=torch.FloatTensor(data.target).unsqueeze(1)  # Convert the target to a
print(X.shape)
print(y.shape)
print(X[0])      # first house's features
print(y[0])      # first house's price 

