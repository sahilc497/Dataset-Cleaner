import logging
import pandas as pd
import sys
import os

def setup_logging():
    """
    Configures logging to both console and a file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("cleaning_log.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("DatasetCleaner")

def load_data(file_path):
    """
    Loads a CSV file into a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

def save_data(df, file_path="output/clean_data.csv"):
    """
    Saves the DataFrame to a CSV file. Creates the directory if it doesn't exist.
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        df.to_csv(file_path, index=False)
        return file_path
    except Exception as e:
        raise Exception(f"Error saving CSV: {e}")

def print_summary(report):
    """
    Prints a formatted summary of the cleaning operations.
    """
    print("\n" + "="*40)
    print("       DATA CLEANING SUMMARY")
    print("="*40)
    for key, value in report.items():
        print(f"{key:25}: {value}")
    print("="*40 + "\n")
