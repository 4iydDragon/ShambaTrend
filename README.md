# Rental Property Prediction

This repository contains code for predicting rental property prices based on various features such as location, number of bedrooms, property type, furnishing, and presence of servants' quarters.

## Project Structure

The project consists of the following files:

- `extraction.py`: Python script for extracting relevant information from property data and preparing it for prediction.
- `rental_predict.py`: Python script for training a deep learning model to predict rental property prices.
- `Property_data_cleaned.csv`: Cleaned property data after extraction and preprocessing.
- `Property_data_output.csv`: Raw property data.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Plotly Express
- Plotly Graph Objects
- TensorFlow
- Keras
- scikit-learn

## Usage

1. Make sure you have all the required dependencies installed.
2. Place the raw property data file (`Property_data_output.csv`) in the same directory as the scripts.
3. Run `extraction.py` to extract relevant information, clean the data, and save it as `Property_data_cleaned.csv`.
4. Run `rental_predict.py` to train the deep learning model and predict rental property prices based on user input.

Please note that the paths used in the code are specific to the author's local machine. You may need to modify the file paths accordingly before running the scripts.

# Rental Property Prediction
This project is designed to predict the rental price of a property based on various features such as the town, number of bedrooms, property type, furnishing status, and whether servants' quarters are included. The prediction is performed using a Long Short-Term Memory (LSTM) neural network.

## File Descriptions
### Data Extraction (`extraction.py`)
This Python script (`extraction.py`) is responsible for extracting relevant information from the raw rental property data and cleaning it for further processing. It contains the following functions:

### extract_bedroom_number(text)
This function takes a text description as input and extracts the number of bedrooms mentioned in the text using regular expressions.

### extract_furnished_data(text)
This function takes a text description as input and extracts the furnishing status (Furnished or Unfurnished) using regular expressions.

## extract_amenity_data(text)
This function takes a text description as input and determines whether servants' quarters are included based on specific keywords (e.g., "sq," "dsq," "Dsq," "SQ," or "DSQ").

### remove_ksh(amount)
This function removes the "Ksh " prefix and any commas from the rental amount to obtain a numerical value.

### remove_percent_encoding(property_type)
This function removes the '%20' characters from the property type to ensure proper labeling.

### rental_predict()
This function runs the Python script rental_predict.py responsible for training the LSTM model and predicting the rental price based on user-provided input.

# Data Cleaning and Preprocessing
## (`rental_predict.py`)
This Python script (`rental_predict.py`) loads the cleaned data from Property_data_cleaned.csv, preprocesses it, trains the LSTM model, and makes predictions. It consists of the following functions:

## Data Preprocessing
The script starts by loading the cleaned data from `Property_data_cleaned.csv`. Then, it converts categorical variables (Property Type, Furnishing, Servants' quarters, and Town) into numerical values using mapping dictionaries.

## LSTM Model
The LSTM model is constructed using Keras Sequential API with the following layers:

1. LSTM layer with 128 units and returning sequences.
2. LSTM layer with 64 units and not returning sequences.
3. Dense layer with 25 units.
4. Output Dense layer with 1 unit.

## Model Training
The model is compiled using the 'adam' optimizer and 'mean_squared_error' as the loss function. It is then trained on the preprocessed data using a batch size of 1 and 2 epochs.

## Model Prediction
After training the model, the script allows users to input specific features (town, bedrooms, property type, furnishing status, and servants' quarters) to predict the rental price for a given property.

## How to Use
1. Ensure that the required Python libraries (e.g., pandas, numpy, matplotlib, plotly, tensorflow, and keras) are installed.
2. Place the raw rental property data in `Property_data_output.csv`.
3. Run the `extraction.py` script to extract, clean, and preprocess the data. The cleaned data will be saved in `Property_data_cleaned.csv`.
4. Run the `rental_predict.py` script to train the LSTM model and perform predictions based on user input.

## Model Description
The Long Short-Term Memory (LSTM) model is a type of recurrent neural network (RNN) that is well-suited for sequence prediction tasks. It is particularly useful when dealing with time-series data or sequences of variable lengths, making it suitable for predicting rental prices based on various property features.

The LSTM model in this project is constructed using Keras, a high-level neural network API in TensorFlow. It takes as input the town, number of bedrooms, property type, furnishing status, and servants' quarters. These features are converted into numerical values and fed into the LSTM layers. The model learns the patterns and relationships in the data during the training process.

To train the model, the data is split into training and testing sets using a 90:10 ratio. The model is trained using the Adam optimizer, which is an extension of stochastic gradient descent, and the mean squared error (MSE) loss function.

After training, the model can make predictions by taking user input for the specific property features. The model predicts the rental price in Kenyan Shillings (Ksh) based on the provided information.

## Author
The project was done by Emmanuel.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](The License is me, ni ulize).
