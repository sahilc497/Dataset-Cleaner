import argparse
import sys
from utils import setup_logging, load_data, save_data, print_summary
from cleaner import DataCleaner, FeatureAnalyzer

def main():
    # Setup CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Dataset Auto Cleaner - Automatically clean your CSV data for ML.")
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", default="output/clean_data.csv", help="Path to save the cleaned CSV (default: output/clean_data.csv).")
    parser.add_argument("-t", "--target", help="Target column for feature importance analysis.")
    parser.add_argument("--task", choices=['classification', 'regression'], default='classification', help="Task type for importance analysis (default: classification).")
    
    args = parser.parse_args()
    
    # Initialize Logger
    logger = setup_logging()
    logger.info("Starting Dataset Auto Cleaner...")
    
    try:
        # Load Data
        logger.info(f"Loading data from: {args.input}")
        df = load_data(args.input)
        
        # Initialize Cleaner
        cleaner = DataCleaner(df)
        
        # Execute Cleaning Pipeline
        logger.info("Cleaning missing values...")
        cleaner.handle_missing_values()
        
        logger.info("Removing duplicate rows...")
        cleaner.remove_duplicates()
        
        logger.info("Removing outliers (IQR method)...")
        cleaner.remove_outliers()
        
        logger.info("Encoding categorical features...")
        cleaner.encode_categorical()
        
        logger.info("Normalizing numerical features...")
        cleaner.normalize_numerical()
        
        # Get Results
        cleaned_df, report = cleaner.get_cleaned_data()
        
        # Feature Importance Analysis (Optional)
        importance_report = None
        if args.target:
            logger.info(f"Analyzing feature importance for target: {args.target}")
            analyzer = FeatureAnalyzer(cleaned_df, args.target, args.task)
            importance_report = analyzer.analyze_importance()
        
        # Save Data
        logger.info(f"Saving cleaned data to: {args.output}")
        save_data(cleaned_df, args.output)
        
        # Print Summary
        print_summary(report, importance_report)
        logger.info("Data cleaning completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
