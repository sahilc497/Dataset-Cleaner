import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Helper to handle numpy/pandas types in JSON
class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(DataEncoder, self).default(obj)

# Load environment variables
load_dotenv()

class AIEngine:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip(' "')
        self.url = "https://api.mistral.ai/v1/chat/completions"
        self.model = "mistral-medium"

    def is_available(self):
        return bool(self.api_key)

    def generate_dataset_summary(self, df, target=None):
        """
        Creates a technical summary of the dataframe for the AI.
        """
        summary = {
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
            "total_rows": len(df),
        }

        if target and target in df.columns:
            if df[target].nunique() < 20:  # Potential categorical/classification
                counts = df[target].value_counts(normalize=True).to_dict()
                summary["class_imbalance"] = counts
                
        return summary

    def get_ai_suggestions(self, summary, target=None, task=None):
        """
        Asks Mistral for data cleaning and model suggestions.
        """
        if not self.is_available():
            return None

        prompt = f"""
        As an expert Data Scientist, analyze this dataset summary and provide cleaning suggestions.
        
        DATASET SUMMARY:
        {json.dumps(summary, indent=2, cls=DataEncoder)}
        
        CONTEXT:
        - Target Variable: {target if target else "Not specified"}
        - ML Task: {task if task else "Not specified"}
        
        PLEASE PROVIDE:
        1. Columns to drop (and why)
        2. Encoding strategy for categorical features
        3. Scaling/Normalization recommendation
        4. Outlier handling strategy
        5. Suggested ML Model
        6. Handling strategy for class imbalance (if applicable)
        
        Format your response as a structured JSON object.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: AI suggestions unavailable ({str(e)})"

    def get_ai_explanation(self, summary_report):
        """
        Generates a human-readable explanation of the cleaning performed.
        """
        if not self.is_available():
            return "AI explanation unavailable. API key missing."

        prompt = f"""
        Explain the following data cleaning operations to a non-technical user in plain English.
        Be concise and professional.
        
        OPERATIONS PERFORMED:
        {json.dumps(summary_report, indent=2, cls=DataEncoder)}
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"AI was unable to generate an explanation: {str(e)}"
