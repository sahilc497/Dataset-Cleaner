import streamlit as st
import pandas as pd
import os
import json
from cleaner import DataCleaner, FeatureAnalyzer
from ai_engine import AIEngine
from io import BytesIO

# Page Config
st.set_page_config(page_title="CleanSight AI", page_icon="🔍", layout="wide")

# App Header
st.title("🔍 CleanSight AI")
st.markdown("### Intelligent Dataset Cleaning & Analysis")

# Sidebar - Settings & AI
st.sidebar.header("🛠️ Settings")
enable_ai = st.sidebar.checkbox("Enable AI Suggestions (Mistral)", value=False)
enable_importance = st.sidebar.checkbox("Feature Importance Analysis", value=True)

ai_engine = AIEngine()
if enable_ai and not ai_engine.is_available():
    st.sidebar.warning("⚠️ Mistral API Key missing in .env")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Preview (Raw)")
        st.dataframe(df.head(10))

    # Configuration
    with st.expander("⚙️ Cleaning Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        target_col = c1.selectbox("Target Column (optional)", [None] + list(df.columns))
        task_type = c2.selectbox("ML Task", ["classification", "regression"])
        
        st.info("The tool will automatically handle missing values, duplicates, and outliers.")

    # AI Section
    if enable_ai and ai_engine.is_available():
        if st.button("✨ Get AI Strategy"):
            with st.spinner("AI is analyzing your data..."):
                summary = ai_engine.generate_dataset_summary(df, target=target_col)
                suggestions = ai_engine.get_ai_suggestions(summary, target=target_col, task=task_type)
                st.session_state['ai_suggestions'] = suggestions
                
        if 'ai_suggestions' in st.session_state:
            st.markdown("#### 🤖 AI Suggested Strategy")
            
            raw_text = st.session_state['ai_suggestions']
            try:
                # Clean up markdown if present
                clean_json = raw_text.strip().strip("```json").strip("```").strip()
                data = json.loads(clean_json)
                
                # Get the root object if nested
                sug = data.get("cleaning_suggestions", data)
                
                # Layout for suggestions
                c1, c2 = st.columns(2)
                
                with c1:
                    st.info("**📋 Columns to Drop**")
                    drop_info = sug.get("columns_to_drop", {})
                    if isinstance(drop_info, dict):
                        cols = drop_info.get("columns", [])
                        reason = drop_info.get("reason", "")
                        if cols:
                            st.write(f"- **Columns:** {', '.join(cols)}")
                            st.write(f"- **Reason:** {reason}")
                        else:
                            for col, r in drop_info.items():
                                st.write(f"- `{col}`: {r}")
                    elif isinstance(drop_info, list):
                        st.write(f"- {', '.join(drop_info)}")
                    else:
                        st.write(drop_info)

                    st.info("**🏷️ Encoding & Scaling**")
                    enc = sug.get("encoding_strategy", "N/A")
                    st.write(f"**Encoding:** {enc if isinstance(enc, str) else list(enc.values())[0]}")
                    
                    scaling = sug.get("scaling_normalization_recommendation", sug.get("scaling_normalization", {}))
                    if isinstance(scaling, dict):
                        st.write(f"**Scaling:** {scaling.get('method', scaling.get('recommendation', 'StandardScaler'))}")
                    else:
                        st.write(f"**Scaling:** {scaling}")

                with c2:
                    st.info("**🤖 Model Suggestion**")
                    model_info = sug.get("suggested_ml_model", {})
                    if isinstance(model_info, dict):
                        st.write(f"**Recommended:** {model_info.get('model', 'N/A')}")
                        st.write(f"*Reason:* {model_info.get('reason', 'N/A')}")
                    else:
                        st.write(model_info)

                    st.info("**⚖️ Imbalance Handling**")
                    imb = sug.get("class_imbalance_handling", {})
                    if isinstance(imb, dict):
                        st.write(f"**Strategy:** {imb.get('strategy', imb.get('reason', 'N/A'))}")
                    else:
                        st.write(imb)

            except Exception as e:
                # Fallback to raw if parsing fails
                st.write(raw_text)
                st.error(f"UI formatting error: {str(e)}")

    # Processing
    if st.button("🚀 Process & Clean Data"):
        with st.spinner("Cleaning in progress..."):
            # Initialize Cleaner
            cleaner = DataCleaner(df, target_column=target_col)
            
            # Run Pipeline
            cleaner.handle_missing_values()
            cleaner.remove_duplicates()
            cleaner.remove_outliers()
            cleaner.encode_categorical()
            cleaner.normalize_numerical()
            
            cleaned_df = cleaner.df
            
            # Importance Analysis
            importance_results = None
            if enable_importance and target_col:
                analyzer = FeatureAnalyzer(cleaned_df, target_col, task_type)
                importance_results = analyzer.analyze()
            
            # Show Results
            with col2:
                st.subheader("Data Preview (Cleaned)")
                st.dataframe(cleaned_df.head(10))
            
            st.markdown("---")
            st.subheader("📈 Cleaning Report")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Duplicates Removed", cleaner.report.get("Duplicates Removed", 0))
            m2.metric("Missing Values Filled", cleaner.report.get("Missing Values Filled", 0))
            m3.metric("Outliers Removed", cleaner.report.get("Outliers Removed", 0))
            m4.metric("Final Rows", len(cleaned_df))
            
            # Importance Chart
            if importance_results:
                st.markdown("#### 📊 Feature Importance")
                imp_df = pd.DataFrame(importance_results["Top Features"], columns=["Feature", "Score"])
                st.bar_chart(imp_df.set_index("Feature"))

            # AI Explanation
            if enable_ai and ai_engine.is_available():
                explanation = ai_engine.get_ai_explanation(cleaner.report)
                st.markdown("#### 📖 AI Summary")
                st.write(explanation)

            # Download
            csv_buffer = BytesIO()
            cleaned_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_buffer.getvalue(),
                file_name="cleansight_data.csv",
                mime="text/csv"
            )

else:
    st.info("Please upload a CSV file to get started.")
    
# Footer
st.markdown("---")
st.markdown("Created with ❤️ by CleanSight AI")
