import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load agriculture produce data (replace 'path/to/agriculture_produce_data.csv' with your file path)
data = pd.read_csv("/c/Users/emman/Documents/Github/ShambaTrend/agriculture_produce_data.csv")

# Map categorical variables to numerical values
data["Crop Type"] = data["Crop Type"].map({
    "Wheat": 1,
    "Rice": 2,
    "Corn": 3,
    "Barley": 4,
    "Potato": 5,
    "Tomato": 6,
    # Add more crop types as needed
})

data["Harvest Season"] = data["Harvest Season"].map({
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Autumn": 3,
    "Winter": 4,
    # Add more harvest seasons as needed
})

# Drop rows with NaN values
data = data.dropna()

# Prepare input features and target variable
x = np.array(data[["Crop Type", "Harvest Season", "OtherFeatures"]])  # Add other features as needed
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
a = int(input("Crop Type (Wheat = 1, Rice = 2, Corn = 3, Barley = 4, Potato = 5, Tomato = 6): "))
b = int(input("Harvest Season (Spring = 1, Summer = 2, Fall = 3, Winter = 4): "))
# Add input prompts for other features as needed
features = np.array([[a, b, ...]])  # Add other features as needed
predicted_yield = model.predict(features)
print("Predicted Yield:", predicted_yield)
