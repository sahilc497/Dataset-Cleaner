import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataCleaner:
    def __init__(self, df, target_column=None):
        self.df = df.copy()
        self.target = target_column
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
        
        # Determine columns to process (exclude target)
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if self.target in num_cols:
            num_cols = num_cols.drop(self.target)
            
        for col in num_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if self.target in cat_cols:
            cat_cols = cat_cols.drop(self.target)
            
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
        if self.target in num_cols:
            num_cols = num_cols.drop(self.target)
            
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
        if self.target in cat_cols:
            cat_cols = cat_cols.drop(self.target)
            
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
        if self.target in num_cols:
            num_cols = num_cols.drop(self.target)
            
        if not num_cols.empty:
            scaler = StandardScaler()
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
            
        self.report["Numerical Columns Normalized"] = len(num_cols)
        return self

    def get_cleaned_data(self):
        self.report["Final Rows"] = len(self.df)
        self.report["Final Columns"] = len(self.df.columns)
        return self.df, self.report

class FeatureAnalyzer:
    def __init__(self, df, target_column, task_type='classification'):
        self.df = df.copy()
        self.target = target_column
        self.task_type = task_type
        self.importance_report = {}

    def analyze(self):
        """
        Calculates feature importance using Random Forest.
        """
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        if self.task_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            # For classification, ensure target is integer
            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                y = LabelEncoder().fit_transform(y)
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X, y)
        importances = model.feature_importances_
        feature_names = X.columns
        
        feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        self.importance_report = {
            "Top Features": feature_importance[:3],
            "Least Features": feature_importance[-3:]
        }
        return self.importance_report
