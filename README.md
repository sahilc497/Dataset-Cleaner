# 🔍 CleanSight AI: Intelligent Dataset Preprocessing

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Latest-red?style=for-the-badge&logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Mistral AI](https://img.shields.io/badge/Mistral%20AI-Powered-orange?style=for-the-badge&logo=mistralai)

**CleanSight AI** is a state-of-the-art data preprocessing and analysis tool that transforms messy raw datasets into ML-ready assets. By combining rule-based cleaning with **Mistral LLM** reasoning, it doesn't just clean your data—it understands it.

---

## 🤖 AI-Powered Intelligence (Mistral)

CleanSight AI integrates the Mistral LLM to act as your personal Data Science assistant.

- **✨ Smart Preprocessing (`--ai`)**: Analyze your data structure and receive a custom strategy for encoding, scaling, and model selection.
- **📖 Human-Readable Reports (`--explain`)**: Automatically translates technical transformations (like IQR outlier removal) into plain-English explanations.
- **⚖️ Imbalance Detection**: Automatically identifies class distribution issues and suggests SMOTE or weighting strategies.

---

## 🖥️ Streamlit Dashboard (Visual Mode)

Prefer a GUI over the terminal? Launch the **CleanSight Dashboard** for an interactive experience.

### Key Features:
- **Side-by-Side Comparison**: View raw and cleaned dataframes simultaneously.
- **Live Metrics**: Track exactly how many rows were dropped, missing values filled, and outliers removed.
- **Importance Visualization**: Interactive bar charts showing feature significance for your ML task.
- **Strategy Sidebar**: Toggle AI features and view Mistral's logic in a structured layout.

**Launch Command:**
```bash
streamlit run app.py
```

---

## 💻 CLI Usage (Power User Mode)

For automation and speed, use the robust CLI engine.

### Quick Start
```bash
# Basic auto-clean
python main.py data.csv

# AI-assisted cleaning with importance analysis
python main.py data.csv --target price --task regression --ai --explain
```

### Command Reference
| Argument | Description | Pro Tip |
| :--- | :--- | :--- |
| `input` | Path to CSV file | Supports relative and absolute paths. |
| `--target` | Your label/target column | Enabling this activates Feature Importance analysis. |
| `--task` | `classification` / `regression` | Helps AI and Random Forest choose the right logic. |
| `--ai` | Enable Mistral suggestions | Requires a `.env` file with `MISTRAL_API_KEY`. |
| `--explain` | Generate plain English report | Saves a detailed summary to `output/explanation.txt`. |
| `-o` | Output directory | Default is `output/`. |

---

## ⚙️ The Cleaning Engine

CleanSight AI follows a rigorous multi-stage pipeline:

1.  **Imputation**: Fills missing numerical data with the **Median** and categorical data with the **Mode**.
2.  **Deduplication**: Identifies and purges identical rows to prevent model bias.
3.  **Outlier Filtering**: Uses the **Interquartile Range (IQR)** method to safely remove anomalies.
4.  **Encoding**: Automatically converts categorical labels into numerical format using **Label Encoding**.
5.  **Scaling**: Standardizes numerical features using **StandardScaler** (protecting the target column).
6.  **Analysis**: Ranks features using **Random Forest** ensembles to pinpoint what actually drives your target.

---

## 🚀 Installation & Setup

### 1. Clone & Enter
```bash
git clone https://github.com/sahilc497/CleanSight-AI.git
cd CleanSight-AI
```

### 2. Environment Setup
**Windows:** `.\setup_venv.ps1`  
**Mac/Linux:**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure AI (Optional)
Create a `.env` file and add your key:
```text
MISTRAL_API_KEY=your_key_here
```

---

## 🧪 Try the Demo
CleanSight AI comes with built-in data generators for testing:
```bash
python generate_sample.py          # Regression Demo
python generate_classification.py  # Classification Demo
```

---

Created with ❤️ by [Sahil](https://github.com/sahilc497)
