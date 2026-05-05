import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.report = {
            "Original Rows": self.original_shape[0],
            "Original Columns": self.original_shape[1]
        }

    def handle_missing_values(self):
        """
        Fill numerical missing values with median and categorical with mode.
        """
        missing_count = self.df.isnull().sum().sum()
        
        # Numerical columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        self.report["Missing Values Filled"] = missing_count
        return self

    def remove_duplicates(self):
        """
        Remove duplicate rows.
        """
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        self.report["Duplicates Removed"] = initial_count - len(self.df)
        return self

    def remove_outliers(self):
        """
        Detect and remove outliers using the IQR method.
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        
        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            total_outliers += len(outliers)
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
        self.report["Outliers Removed"] = total_outliers
        return self

    def encode_categorical(self):
        """
        Encode categorical variables using Label Encoding.
        """
        cat_cols = self.df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in cat_cols:
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            
        self.report["Categorical Columns Encoded"] = len(cat_cols)
        return self

    def normalize_numerical(self):
        """
        Normalize numerical features using StandardScaler.
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            scaler = StandardScaler()
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
            
        self.report["Numerical Columns Normalized"] = len(num_cols)
        return self

    def get_cleaned_data(self):
        self.report["Final Rows"] = len(self.df)
        self.report["Final Columns"] = len(self.df.columns)
        return self.df, self.report
