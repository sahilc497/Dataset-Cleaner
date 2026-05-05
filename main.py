import argparse
import pandas as pd
import os
import logging
from cleaner import DataCleaner, FeatureAnalyzer
from utils import setup_logging, save_data, print_summary
from ai_engine import AIEngine

def main():
    parser = argparse.ArgumentParser(description="CleanSight AI - Intelligent Dataset Cleaner")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save output (default: output/)")
    parser.add_argument("--target", help="Target column for analysis")
    parser.add_argument("--task", choices=["classification", "regression"], help="ML Task type")
    
    # AI Arguments
    parser.add_argument("--ai", action="store_true", help="Enable AI suggestions from Mistral")
    parser.add_argument("--explain", action="store_true", help="Generate human-readable explanation of cleaning")
    parser.add_argument("--report", action="store_true", help="Generate a comprehensive AI report")

    args = parser.parse_args()
    logger = setup_logging()
    ai_engine = AIEngine()

    try:
        logger.info("Starting CleanSight AI...")
        
        # Load Data
        df = pd.read_csv(args.input)
        
        # AI Suggestions (Pre-Cleaning)
        if args.ai:
            if not ai_engine.is_available():
                logger.error("Mistral API Key missing! Set MISTRAL_API_KEY in .env file.")
            else:
                logger.info("Fetching AI suggestions from Mistral...")
                data_summary = ai_engine.generate_dataset_summary(df, target=args.target)
                suggestions = ai_engine.get_ai_suggestions(data_summary, target=args.target, task=args.task)
                
                print("\n" + "="*40)
                print("       AI PREPROCESSING SUGGESTIONS")
                print("="*40)
                print(suggestions)
                print("="*40 + "\n")
                
                # Save suggestions
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, "ai_suggestions.json"), "w") as f:
                    f.write(str(suggestions))

        # Initialize Cleaner
        cleaner = DataCleaner(df, target_column=args.target)
        
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

        # Feature Importance Analysis
        analysis_report = None
        if args.target:
            logger.info(f"Analyzing feature importance for target: {args.target}")
            analyzer = FeatureAnalyzer(cleaner.df, args.target, args.task)
            analysis_report = analyzer.analyze()

        # Save Cleaned Data
        final_output_path = os.path.join(args.output_dir, "clean_data.csv")
        save_data(cleaner.df, final_output_path)
        logger.info(f"Saving cleaned data to: {final_output_path}")

        # Print Summary
        print_summary(cleaner.report, analysis_report)

        # AI Explanation (Post-Cleaning)
        if args.explain:
            if not ai_engine.is_available():
                logger.warning("AI explanation skipped: API key missing.")
            else:
                logger.info("Generating AI explanation...")
                explanation = ai_engine.get_ai_explanation(cleaner.report)
                print("\nAI EXPLANATION:")
                print(explanation)
                
                with open(os.path.join(args.output_dir, "explanation.txt"), "w") as f:
                    f.write(explanation)

        logger.info("Data cleaning completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
