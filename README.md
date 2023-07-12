# <b>Parkinson Disease Detection </b>
This project aims to detect Parkinson's disease using machine learning techniques. The dataset used for training and testing the model is imported from a CSV file.

# Dependencies
This project requires the following libraries:

numpy
pandas
scikit-learn (sklearn)
Make sure these libraries are installed in your Python environment.

#Usage
Import the necessary libraries:

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn.metrics import accuracy_score

#Load the dataset:


data = pd.read_csv('data.csv')

# Check for any missing values in the dataset and handle them using mean, median, or mode:

data.fillna(data.mean(), inplace=True)  # Replace missing values with mean

Split the dataset into input features and target variable:


X = data.drop(['name', 'status'], axis=1)  # Input features

y = data['status']  # Target variable

# Split the data into training and testing sets:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using the StandardScaler:

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Train the model using Support Vector Machine (SVM) classifier:

model = svm.SVC()

model.fit(X_train, y_train)

# Evaluate the accuracy of the trained model:

train_accuracy = accuracy_score(y_train, model.predict(X_train))

test_accuracy = accuracy_score(y_test, model.predict(X_test))


print("Train accuracy:", train_accuracy)

print("Test accuracy:", test_accuracy)

# Predict the presence of Parkinson's disease for new input data:

new_data = np.array([[...]])  # Input features for prediction

new_data = scaler.transform(new_data)

prediction = model.predict(new_data)


if prediction == 0:

    print("The person is healthy.")
    
else:

    print("The person has Parkinson's disease.")
    

# Project View


