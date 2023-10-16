import pandas as pd
import numpy as np
import re
import subprocess

def extract_crop_type(row):
    text = row["Information"]  # Assuming crop type information is in the "Information" column
    # Add regex pattern for extracting crop type from the text
    pattern = r'(wheat|rice|corn|barley|potato|tomato|...)'  # Add more crop types as needed
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()  # Capitalize the crop type for consistency
    else:
        return "Unknown Crop"

def extract_harvest_season(row):
    text = row["Information"]  # Assuming harvest season information is in the "Information" column
    # Add regex pattern for extracting harvest season from the text
    pattern = r'(spring|summer|fall|autumn|winter)'  # Add more harvest seasons as needed
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()  # Capitalize the harvest season for consistency
    else:
        return "Unknown Season"

def extract_yield(row):
    text = row["Information"]  # Assuming yield information is in the "Information" column
    # Add regex pattern for extracting yield from the text
    pattern = r'(\d+\.?\d*)\s*(?:tons|kg|pounds|bushels)'  # Add more units as needed
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))  # Convert yield to float for numerical value
    else:
        return np.nan  # Return NaN for missing yield information

def agriculture_produce_extraction():
    data = pd.read_csv("/c/Users/emman/Documents/Github/ShambaTrend/agriculture_produce_data.csv")  # Replace with your agriculture produce data file path

    # Add new columns for the extracted crop type, harvest season, and yield data.
    data["Crop Type"] = data.apply(extract_crop_type, axis=1)
    data["Harvest Season"] = data.apply(extract_harvest_season, axis=1)
    data["Yield"] = data.apply(extract_yield, axis=1)

    # Remove null records in the Yield column.
    data = data.dropna(subset=["Yield"])

    # Save the changes to a new CSV file without including unnecessary columns.
    data = data.drop(["Information", "OtherColumnsToRemove"], axis=1)  # Add other unnecessary columns to remove
    data.to_csv("/path/to/your/agriculture_produce_cleaned.csv", index=False)  # Replace with your desired output file path

    # Call any additional functions or processes related to agriculture produce extraction here
    # ...

agriculture_produce_extraction()
