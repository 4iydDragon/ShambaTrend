import pandas as pd
import numpy as np
import re
import subprocess

def extract_region(row):
    text = row["Region of Farming"]  # Assuming harvest season information is in the "Information" column
    # Add regex pattern for extracting harvest season from the text
    pattern = r'(rift valley|central|nairobi|eastern|north eastern|coast|western|nyanza)'  # Add more harvest seasons as needed
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()  # Capitalize the harvest season for consistency
    else:
        return "Unknown Size"
    
def extract_crop_type(row):
    text = str(row["Type of Crops"])  # Convert the input to a string
    pattern = r'(maize|beans|vegetables|onions|potatoes|tomatoes)'  # Add more crop types as needed
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()  # Capitalize the crop type for consistency
    else:
        return "Unknown Crop"


def extract_size_of_field(row):
    text = str(row["Size of Farm"])  # Assuming harvest season information is in the "Information" column
    # Add regex pattern for extracting harvest season from the text
    pattern = r'(small|medium|large)'  # Add more harvest seasons as needed
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()  # Capitalize the harvest season for consistency
    else:
        return "Unknown Size"

def extract_harvest_season(row):
    text = row["Harvest Season"]  # Assuming harvest season information is in the "Information" column
    # Add regex pattern for extracting harvest season from the text
    pattern = r'(spring|summer|fall|autumn|winter)'  # Add more harvest seasons as needed
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()  # Capitalize the harvest season for consistency
    else:
        return "Unknown Season"

def extract_yield(row):
    value = row["Yield"]
    if isinstance(value, (int, float)):
        return value  # Return the numerical value as is
    else:
        text = str(value)  # Convert the input to a string
        pattern = r'(\d+\.?\d*)'  # This pattern captures numerical values
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))  # Convert yield to float for numerical value
        else:
            return np.nan  # Return NaN for missing yield information


def agriculture_produce_extraction():
    """Runs the produce_predict python file"""
    script_path = 'C:/Users/emman/Documents/Github/ShambaTrend/produce_predict.py'
    subprocess.run(['python3', script_path])

data = pd.read_csv("C:/Users/emman/Documents/Github/ShambaTrend/Responses.csv")  # Replace with your agriculture produce data file path

# Add new columns for the extracted crop type, harvest season, and yield data.
    
data["Region of Farming"] = data.apply(extract_region, axis=1)
data["Type of Crops"] = data.apply(extract_crop_type, axis=1)
data["Size of Farm"] = data.apply(extract_size_of_field, axis=1)
data["Harvest Season"] = data.apply(extract_harvest_season, axis=1)
data["Yield"] = data.apply(extract_yield, axis=1)

# Remove null records in the Yield column.
data = data.dropna(subset=["Yield"])

# Save the changes to a new CSV file without including unnecessary columns.
data = data.drop(["Timestamp", "Name", "Phone", "2.1.", "2.2.", "3.1.", "3.2.", "4.2.", "4.1.", "5.1.", "5.2."], axis=1)
data.to_csv("C:/Users/emman/Documents/Github/ShambaTrend/agriculture_produce_cleaned.csv", index=False)  # Replace with your desired output file path

# Call any additional functions or processes related to agriculture produce extraction here
# ...

agriculture_produce_extraction()
