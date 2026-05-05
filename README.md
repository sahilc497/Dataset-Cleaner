# 🚀 Dataset Auto Cleaner

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Latest-red?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?style=for-the-badge&logo=scikitlearn)

A powerful, CLI-based automation tool designed to streamline the data preprocessing phase for Machine Learning projects. It handles missing values, duplicates, outliers, and feature scaling with a single command.

---

## 🌟 Key Features

- **🧠 Intelligent Imputation**: 
  - Numerical data: Fills missing values with the **Median**.
  - Categorical data: Fills missing values with the **Mode**.
- **🧹 Data Deduplication**: Removes redundant rows to ensure data integrity.
- **📉 Outlier Management**: Detects and removes outliers using the **Interquartile Range (IQR)** method.
- **🏷️ Categorical Encoding**: Automatically converts text labels into numerical format using **Label Encoding**.
- **⚖️ Feature Scaling**: Standardizes numerical features using **StandardScaler** for better ML performance.
- **📁 Organized Output**: Automatically creates an `output/` directory for cleaned datasets.
- **📜 Detailed Logging**: Maintains a full audit trail of cleaning operations in `cleaning_log.log`.

---

## 🛠️ Project Structure

```text
├── output/             # Cleaned datasets are saved here
├── venv/               # Virtual environment (ignored by git)
├── main.py             # CLI entry point & workflow orchestration
├── cleaner.py          # Core data cleaning engine
├── utils.py            # I/O utilities and logging setup
├── requirements.txt    # Project dependencies
└── README.md           # Professional documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sahilc497/Dataset-Cleaner.git
cd Dataset-Cleaner
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
