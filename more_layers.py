from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# Call the functions
collect_data()
preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)
train_models(X_train, y_train)


# Function to train models
def train_models(X_train, y_train):
    # Model 1: Linear Regression
    model1 = LinearRegression()
    model1.fit(X_train, y_train)

    # Model 2: Decision Tree Regressor
    model2 = DecisionTreeRegressor()
    model2.fit(X_train, y_train)

    # Model 3: Random Forest Regressor
    model3 = RandomForestRegressor()
    model3.fit(X_train, y_train)

    # Model 4: Support Vector Regressor
    model4 = SVR()
    model4.fit(X_train, y_train)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color="blue", label="Actual")
    plt.plot(X_train, model1.predict(X_train), color="red", label="Linear Regression")
    plt.plot(X_train, model2.predict(X_train), color="green", label="Decision Tree")
    plt.plot(X_train, model3.predict(X_train), color="orange", label="Random Forest")
    plt.plot(X_train, model4.predict(X_train), color="purple", label="Support Vector")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Regression Models")
    plt.legend()
    plt.show()


# Call the functions
collect_data()
preprocess_data()
X_train, X_test, y_train, y_test = split_data(X, y)
train_models(X_train, y_train)
live_inference()


# Function for live inferencing
def live_inference():
    cap = cv2.VideoCapture(0)  # Open the default camera
    while True:
        ret, frame = cap.read()  # Read the frame from the camera
        # Preprocess the frame if needed
        # Perform inference using the trained models
        # Display the results on the frame
        cv2.imshow("Live Inference", frame)  # Display the frame with results
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit if 'q' is pressed
            break
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows


# Call the functions
collect_data()
preprocess_data()
X_train, y_train = split_data()
train_models(X_train, y_train)
live_inference()
