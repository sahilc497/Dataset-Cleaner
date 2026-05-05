# 🔍 CleanSight AI

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Latest-red?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?style=for-the-badge&logo=scikitlearn)

A powerful, CLI-based automation tool designed to streamline the data preprocessing phase for Machine Learning projects. Now supercharged with **Mistral AI**.

---

## 🤖 AI-Powered Features (Mistral)

CleanSight AI now leverages the **Mistral AI** engine to provide intelligent suggestions and explanations for your data cleaning process.

### 🔑 Setup AI
1. Get a Mistral API Key from [console.mistral.ai](https://console.mistral.ai/).
2. Create a `.env` file in the root directory:
   ```text
   MISTRAL_API_KEY=your_api_key_here
   ```

### 🤖 AI Commands
- **`--ai`**: Fetches smart preprocessing suggestions (columns to drop, scaling strategies, model recommendations).
- **`--explain`**: Generates a plain-English explanation of all cleaning steps performed.

**Example Usage:**
```bash
python main.py data.csv --target status --task classification --ai --explain
```

## 🚀 Features

- **Automated Cleaning**:
  - Handles missing values (Median for Numerical, Mode for Categorical).
  - Removes duplicate rows.
  - Detects and removes outliers using the **IQR (Interquartile Range)** method.
- **Preprocessing**:
  - **Label Encoding**: Automatically encodes categorical text data.
  - **Standardization**: Normalizes numerical features (scales to mean=0, std=1).
  - **Target Protection**: Ensures the target column is never scaled or modified.
- **Smart Analysis**:
  - **Feature Importance**: Uses Random Forest to identify the most and least significant features.
  - **Dual Task Support**: Works for both **Regression** (numerical target) and **Classification** (categorical target).

## 🛠️ Usage

### 🖥️ Streamlit UI (New!)
Prefer a visual interface? Run our dashboard:
```bash
streamlit run app.py
```

### 💻 CLI Usage
```bash
# Basic Cleaning
python main.py data.csv

# Full AI Analysis
python main.py data.csv --target price --task regression --ai --explain
```

### CLI Arguments
- `input`: Path to your CSV file.
- `--output_dir`: Custom folder for results (default: `output/`).
- `--target`: The name of your target/label column.
- `--task`: `classification` or `regression`.

---

## 🧪 Try it Now (Demo)

Don't have a dataset? Use our built-in synthetic data generators to see the tool in action:

1.  **Generate Synthetic Data**:
    ```bash
    python generate_sample.py          # Creates synthetic_data.csv (Regression)
    python generate_classification.py  # Creates classification_data.csv (Classification)
    ```

2.  **Run Analysis**:
    ```bash
    # Test Regression
    python main.py synthetic_data.csv --target target --task regression

    # Test Classification
    python main.py classification_data.csv --target status --task classification
    ```

---

## 🛠️ Project Structure

```text
├── output/                  # Cleaned datasets are saved here
├── app.py                   # Streamlit Web Dashboard
├── main.py                  # CLI entry point & workflow orchestration
├── cleaner.py               # Core data cleaning & analysis engine
├── ai_engine.py             # Mistral AI integration logic
├── utils.py                 # I/O utilities and logging setup
├── requirements.txt         # Project dependencies
├── generate_sample.py       # (Demo) Script for regression data
├── generate_classification.py # (Demo) Script for classification data
└── README.md                # Professional documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sahilc497/CleanSight-AI.git
cd CleanSight-AI
```

### 2. Setup Environment

**Windows (PowerShell):**
```powershell
.\setup_venv.ps1
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Cleaner
Provide the path to your raw CSV file:
```bash
python main.py data.csv
```

---

## ⚙️ Advanced Usage

You can specify a custom output path using the `-o` or `--output` flag:

```bash
python main.py raw_data.csv -o output/final_cleaned_data.csv
```

---

## 📊 Feature Importance Analysis
Identify which features contribute most to your target variable:
```bash
python main.py data.csv --target salary --task regression
```

---

## 📊 Summary Report
After every run, the tool generates a console summary:
```text
========================================
       DATA CLEANING SUMMARY
========================================
Original Rows            : 1000
Original Columns         : 12
Missing Values Filled    : 45
Duplicates Removed       : 5
Outliers Removed         : 12
Categorical Columns Encoded: 3
Numerical Columns Normalized: 9
Final Rows               : 983
Final Columns            : 12
========================================
```

---

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
Created with ❤️ by [Sahil](https://github.com/sahilc497)
