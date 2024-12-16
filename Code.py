import sklearn
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import kagglehub

# Load Kaggle Dataset after local download
# Dataset used: Swedish Auto Insurance Dataset
path = kagglehub.dataset_download("redwankarimsony/auto-insurance-in-sweden")
data = pd.DataFrame({ 'claims': [108, 19, 13, 124, 40, 57, 23, 14, 45, 10, 5, 48, 11, 23, 7, 2, 24, 6, 3, 23, 6, 9, 9, 3, 29, 7, 4, 20, 7, 4, 0, 25, 6, 5, 22, 11, 61, 12, 4, 16, 13, 60, 41, 37, 55, 41, 11, 27, 8, 3, 17, 13, 13, 15, 8, 29, 30, 24, 9, 31, 14, 53, 26],
                     'payment': [392.5, 46.2, 15.7, 422.2, 119.4, 170.9, 56.9, 77.5, 214, 65.3, 20.9, 248.1, 23.5, 39.6, 48.8, 6.6, 134.9, 50.9, 4.4, 113, 14.8, 48.7, 52.1, 13.2, 103.9, 77.5, 11.8, 98.1, 27.9, 38.1, 0, 69.2, 14.6, 40.3, 161.5, 57.2, 217.6, 58.1, 12.6, 59.6, 89.9, 202.4, 181.3, 152.8, 162.8, 73.4, 21.3, 92.6, 76.1, 39.9, 142.1, 93, 31.9, 32.1, 55.6, 133.3, 194.5, 137.9, 87.4, 209.8, 95.5, 244.6, 187.5]
                    })
print(data.head())


x = data[['claims']]
y = data['payment']


# Create test and training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Convert to Tensors
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)


#Understand your data
print(len(x_train), len(y_train), len(x_test), len(y_test))
print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(x_test))


#Visualize your data
def plot_predictions(train_data=x_train, train_labels=y_train, test_data=x_test, test_labels=y_test, predictions=None):
  plt.figure( figsize=(8, 6))
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
  plt.legend()


plot_predictions()


model_0 = LinearRegression().fit(x_train, y_train)


# Cross-Validation technique
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(model_0, x_train_tensor, y_train_tensor, cv=kf, scoring='neg_mean_absolute_error')
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores)}")


# Bootstrap Technique
bootstrap_scores = []
epochs = 10  # Number of bootstrap iterations


for epoch in range(epochs):
    # Resample the training data with replacement
    x_train_bootstrap, y_train_bootstrap = resample(x_train_tensor, y_train_tensor)


    # Train the model
    model_1 = LinearRegression().fit(x_train_bootstrap, y_train_bootstrap)


    y_pred = model_1.predict(x_test)
    bootstrap_scores.append(np.mean(np.abs(y_test - y_pred)))


print(f"Bootstrap Scores: {bootstrap_scores}")
print(f"Mean Bootstrap Score: {np.mean(bootstrap_scores)}")


# Plot Predictions
plot_predictions(predictions=y_pred)
