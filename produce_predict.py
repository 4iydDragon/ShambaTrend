import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load agriculture produce data (replace 'path/to/agriculture_produce_data.csv' with your file path)
data = pd.read_csv("C:/Users/emman/Documents/Github/ShambaTrend/agriculture_produce_cleaned.csv")

# Map categorical variables to numerical values

data["Region of Farming"] = data["Region of Farming"].map({
    "Rift Valley": 1,
    "Central": 2,
    "Nairobi": 3,
    "Eastern": 4,
    "North Eastern": 5,
    "Coast": 6,
    "Western": 7,
    "Nyanza": 8
    # Add more regions as needed
})

data["Type of Crops"] = data["Type of Crops"].map({
    "Maize": 1,
    "Beans": 2,
    "Vegetables": 3,
    "Onions": 4,
    "Potatoes": 5,
    "Tomatooes": 6
    # Add more crop types as needed
})

data["Size of Farm"] = data["Size of Farm"].map({
    "Small": 1,
    "Medium": 2,
    "Large": 3
    # Add more sizes as needed
})

data["Harvest Season"] = data["Harvest Season"].map({
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Autumn": 5,
    "Winter": 6
    # Add more harvest seasons as needed
})

# Drop rows with NaN values
data = data.dropna()

# Prepare input features and target variable
x = np.array(data[["Region of Farming", "Type of Crops", "Size of Farm", "Harvest Season"]])  # Add other features as needed
y = np.array(data[["Yield"]])

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.summary()

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=21)

# Predict agriculture produce yield
print("Enter Agriculture Produce Details to Predict Yield")

a = int(input("Region of Farming (Rift Valley= 1, Central= 2, Nairobi= 3, Eastern= 4, North Eastern= 5, Coast= 6, Western= 7, Nyanza= 8: "))
b = int(input("Type of Crops (Maize = 1, Beans = 2, Vegetables = 3, Onions = 4, Potatoes = 5, Tomatoes = 6): "))
c = int(input("Size of Farm (Small = 1, Medium = 2, Large = 3): "))
d = int(input("Harvest Season (Spring = 1, Summer = 2, Fall = 3, Winter = 4): "))

# Add input prompts for other features as needed
features = np.array([[a, b, c, d ]])  # Add other features as needed
predicted_yield = model.predict(features)
print("Predicted Yield:", predicted_yield)
