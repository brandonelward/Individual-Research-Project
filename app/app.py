import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score, PrecisionRecallDisplay, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, RocCurveDisplay
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Preprocessing Functions
def preprocess_transactions_data(df, for_performance=False):
    """Prepares the kaggle PaySim dataset by one-hot encoding and creating error features."""
    df_processed = df.copy()
    
    # Defines the final features the model needs
    model_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'errorBalanceOrig', 'errorBalanceDest']
    
    # One-hot encodes the 'type' column
    if 'type' in df_processed.columns:
        type_dummies = pd.get_dummies(df_processed['type'], prefix='type', drop_first=True)
        df_processed = pd.concat([df_processed, type_dummies], axis=1)

    # Engineer features related to balance discrepancies
    df_processed['errorBalanceOrig'] = df_processed.get('oldbalanceOrg', 0) - df_processed.get('amount', 0) - df_processed.get('newbalanceOrig', 0)
    df_processed['errorBalanceDest'] = df_processed.get('oldbalanceDest', 0) + df_processed.get('amount', 0) - df_processed.get('newbalanceDest', 0)

    # Ensures all model features exist, fill with 0 if not
    for col in model_features:
        if col not in df_processed.columns:
            df_processed[col] = 0
            
    if for_performance and 'isFraud' in df.columns:
        X = df_processed[model_features]
        y = df['isFraud']
        return X, y
    else:
        if 'isFraud' in df_processed.columns:
            df_processed = df_processed.drop(columns=['isFraud'])
        return df_processed[model_features]

def preprocess_credit_card_data(df, for_performance=False):
    """Prepares the Kaggle Credit Card dataset."""
    df_processed = df.copy()
    
    # Defines features
    model_features = [f'V{i}' for i in range(1, 29)] + ['scaled_Amount']
    
    scaler = StandardScaler()
    if 'Amount' in df_processed.columns:
        df_processed['scaled_Amount'] = scaler.fit_transform(df_processed['Amount'].values.reshape(-1, 1))

    for col in model_features:
        if col not in df_processed.columns:
            df_processed[col] = 0

    if for_performance and 'Class' in df.columns:
        X = df_processed[model_features]
        y = df['Class']
        return X, y
    else:
        if 'Class' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Class'])
        return df_processed[model_features]

# Model Comparison Functions
@st.cache_resource
def load_all_models():
    """Load all models and explainers for comparison."""
    script_dir = Path(__file__).parent.parent
    
    models = {}
    explainers = {}
    
    # Credit Card Models
    models['Credit Card - Ensemble'] = joblib.load(script_dir / "models" / "creditcard_ensemble_model.joblib")
    models['Credit Card - XGBoost'] = joblib.load(script_dir / "models" / "xgboost_creditcard_model.joblib")
    explainers['Credit Card - Ensemble'] = joblib.load(script_dir / "models" / "shap_explainer_ensemble_creditcard.joblib")
    explainers['Credit Card - XGBoost'] = joblib.load(script_dir / "models" / "shap_explainer_xgboost_creditcard.joblib")
    
    # Transaction Models
    models['Transactions - Ensemble'] = joblib.load(script_dir / "models" / "ensemble_transactions_model.joblib")
    models['Transactions - XGBoost'] = joblib.load(script_dir / "models" / "xgboost_transactions_model.joblib")
    explainers['Transactions - Ensemble'] = joblib.load(script_dir / "models" / "shap_explainer_ensemble_transactions.joblib")
    explainers['Transactions - XGBoost'] = joblib.load(script_dir / "models" / "shap_explainer_xgboost_transactions.joblib")
    
    return models, explainers

def calculate_model_metrics(model, X, y):
    """Calculates comprehensive metrics for a model."""
    
    y_pred = model.predict(X)
    y_scores = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_scores),
        'auprc': average_precision_score(y, y_scores),
        'predictions': y_pred,
        'probabilities': y_scores
    }
    
    return metrics

def create_comparison_visualisations(metrics_dict, dataset_name):
    """Creates interactive comparison visualisations using Plotly."""
    
    # Prepares data for visualization
    models = list(metrics_dict.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'auprc']
    
    # Creates subplot figure
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'AUPRC'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for Ensemble, Orange for XGBoost
    
    for i, metric in enumerate(metrics_names):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        values = [metrics_dict[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric.upper(),
                marker_color=colors,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=f'Model Performance Comparison - {dataset_name}',
        height=600,
        template='plotly_dark'
    )
    
    return fig

def create_roc_pr_comparison(metrics_dict, X, y, dataset_name):
    """Creates ROC and PR curve comparisons."""
    
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ROC Curves', 'Precision-Recall Curves')
    )
    
    colors = ['#1f77b4', '#ff7f0e']
    
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, metrics['probabilities'])
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC: {metrics["roc_auc"]:.3f})',
                line=dict(color=colors[i], width=2)
            ),
            row=1, col=1
        )
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y, metrics['probabilities'])
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name} (AUPRC: {metrics["auprc"]:.3f})',
                line=dict(color=colors[i], width=2),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Adds baseline lines
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                  line=dict(dash='dash', color='gray'), 
                  name='Random Classifier', showlegend=True),
        row=1, col=1
    )
    
    baseline_pr = y.sum() / len(y)
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[baseline_pr, baseline_pr], mode='lines',
                  line=dict(dash='dash', color='gray'), 
                  name=f'No Skill (PR: {baseline_pr:.3f})', showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'ROC and Precision-Recall Curves - {dataset_name}',
        height=400,
        template='plotly_dark'
    )
    
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    
    return fig

# Cached function to load data efficiently
@st.cache_data
def load_sample_data(path):
    """Loads the sample data from a CSV file and caches it."""
    return pd.read_csv(path)

# Helper function for SHAP plots 
def st_shap_plot(shap_explanation_object):
    """Creates a SHAP waterfall plot and displays it in Streamlit."""
    if len(shap_explanation_object.shape) == 2:
        explanation_to_plot = shap_explanation_object[:, 1]
    else:
        explanation_to_plot = shap_explanation_object
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0E1117")
    shap.waterfall_plot(explanation_to_plot, show=False)
    plt.tight_layout()
    ax.tick_params(colors='white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.ylabel("Feature", color="white", fontsize=12)
    plt.xlabel("SHAP value (impact on model output for 'Fraud' class)", color="white", fontsize=12)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    
    # Adds a comprehensive explanation for better understanding
    st.markdown("""
    ### 📖 **How to Read This SHAP Explanation**
    
    **🎯 What is SHAP?**  
    SHAP (SHapley Additive exPlanations) shows exactly **why** the model made its prediction by breaking down each feature's contribution.
    
    **📊 Reading the Waterfall Plot:**
    - **Starting Point (E[f(X)])**: The average prediction across all transactions (baseline)
    - **Red Bars (→)**: Features that **INCREASE** the fraud probability (push toward fraud)
    - **Blue Bars (←)**: Features that **DECREASE** the fraud probability (push toward legitimate)
    - **Ending Point (f(x))**: The final prediction for this specific transaction
    
    **🔍 Key Insights:**
    - **Longer bars** = stronger influence on the prediction
    - **Feature values** are shown in brackets [value=X.XX]
    - The **sum of all contributions** leads from baseline to final prediction
    - **Most suspicious features** appear at the top with longer red bars
    
    **💡 Business Interpretation:**
    - If final prediction >= 0.5: Model predicts **FRAUD** 🚨
    - If final prediction < 0.5: Model predicts **LEGITIMATE** ✅
    - Red features explain "**Why flagged as fraud**"
    - Blue features explain "**Why NOT flagged as fraud**"
    """)
    
    # Adds a helpful tip box
    st.info("""
    💡 **Pro Tip**: Focus on the **top 3-5 features** with the longest bars - these are the main reasons 
    behind the model's decision. This helps fraud analysts understand which transaction patterns to investigate first!
    """)

# Streamlit App Configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 Explainable AI Fraud Detection System")
st.write("This application uses machine learning to detect fraudulent transactions. Choose a model from the sidebar and an analysis mode below to begin.")

# Sidebar with Model Comparison Option
st.sidebar.header("⚙️ Configuration")

# Adds mode selection
analysis_mode = st.sidebar.radio(
    "Choose Analysis Mode:",
    ("Single Model Analysis", "Model Comparison")
)

if analysis_mode == "Single Model Analysis":
    # Original single model selection
    model_choice = st.sidebar.selectbox(
        "Choose a Detection Model:",
        (
            "Credit Card - Ensemble", 
            "Credit Card - XGBoost",
            "Transactions - Ensemble",
            "Transactions - XGBoost"
        )
    )
    
    # Original artifact loading for single model
    @st.cache_resource
    def load_artifacts(choice):
        """Loads the selected model and SHAP explainer from disk."""
        script_dir = Path(__file__).parent.parent
        
        if "Ensemble" in choice and "Credit Card" in choice:
            model_path = script_dir / "models" / "creditcard_ensemble_model.joblib"
            explainer_path = script_dir / "models" / "shap_explainer_ensemble_creditcard.joblib"
        elif "XGBoost" in choice and "Credit Card" in choice:
            model_path = script_dir / "models" / "xgboost_creditcard_model.joblib"
            explainer_path = script_dir / "models" / "shap_explainer_xgboost_creditcard.joblib"
        elif "Ensemble" in choice and "Transactions" in choice:
            model_path = script_dir / "models" / "ensemble_transactions_model.joblib"
            explainer_path = script_dir / "models" / "shap_explainer_ensemble_transactions.joblib"
        else: # XGBoost Transactions
            model_path = script_dir / "models" / "xgboost_transactions_model.joblib"
            explainer_path = script_dir / "models" / "shap_explainer_xgboost_transactions.joblib"
        
        model = joblib.load(model_path)
        explainer = joblib.load(explainer_path)
        return model, explainer

    # Sets up paths and functions based on choice
    if "Transactions" in model_choice:
        preprocessing_function = preprocess_transactions_data
        sample_data_path = Path(__file__).parent.parent / "data" / "transactions_sample_280k.csv"
        ground_truth_col = 'isFraud'
        if "Ensemble" in model_choice:
            st.sidebar.info("This is an ensemble model (RF & LR) trained on the PaySim transactions data.")
        else:
            st.sidebar.info("This XGBoost model is trained on PaySim transactions data.")
    else:
        preprocessing_function = preprocess_credit_card_data
        sample_data_path = Path(__file__).parent.parent / "data" / "creditcard.csv"
        ground_truth_col = 'Class'
        if "Ensemble" in model_choice:
            st.sidebar.info("This is a weighted ensemble (RF & LR) trained on real, anonymised credit card data.")
        else:
            st.sidebar.info("This XGBoost model is trained on the same real, anonymised credit card data.")

    # Loads artifacts and data for single model
    try:
        model, explainer = load_artifacts(model_choice)
        sample_df = load_sample_data(sample_data_path)
        max_rows = len(sample_df)
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please ensure all model and data files are in the correct directories.")
        st.info(f"Missing file details: {e}")
        st.stop()

    if 'alerts' not in st.session_state:
        st.session_state.alerts = []

    # TABS FOR SINGLE MODEL ANALYSIS
    tab1, tab2, tab3 = st.tabs(["📂 **Batch Analysis**", "⏱️ **Real-Time Simulation**", "📊 **Model Performance & Explainability**"])

    # Batch Analysis Tab
    with tab1:
        st.header("Analyse a File of Transactions")
        uploaded_file = st.file_uploader("Upload a CSV file compatible with the selected model:", type="csv", key=f"uploader_{model_choice}")
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.write("Preview of your uploaded transactions:")
            st.dataframe(user_data.head())
            
            if st.button("🔍 Analyse Batch File", type="primary"):
                with st.spinner(f'Analysing {len(user_data)} transactions with {model_choice}...'):
                    processed_data = preprocessing_function(user_data, for_performance=False)
                    predictions = model.predict(processed_data)
                    user_data['Fraud Prediction'] = predictions
                    flagged_transactions = user_data[user_data['Fraud Prediction'] == 1]
                    
                    shap_values_flagged = None
                    if not flagged_transactions.empty:
                        flagged_processed_data = processed_data.loc[flagged_transactions.index]
                        raw_shap_values = explainer.shap_values(flagged_processed_data)
                        shap_values_flagged = raw_shap_values[1] if isinstance(raw_shap_values, list) else raw_shap_values

                st.success("Analysis Complete!")
                if not flagged_transactions.empty:
                    st.warning(f"🚨 Found {len(flagged_transactions)} potentially fraudulent transaction(s)!")
                    for i, (index, row) in enumerate(flagged_transactions.iterrows()):
                        st.subheader(f"Transaction #{index} (from original file)")
                        st.dataframe(pd.DataFrame(row).transpose())
                        with st.expander("🔍 See why this transaction was flagged (SHAP Explanation)"):
                            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                            shap_explanation = shap.Explanation(
                                values=shap_values_flagged[i], 
                                base_values=expected_value, 
                                data=flagged_processed_data.iloc[i], 
                                feature_names=processed_data.columns.tolist()
                            )
                            st_shap_plot(shap_explanation)
                else:
                    st.success("✅ Great news! No suspicious transactions were detected in this file.")

    # Real-Time Simulation Tab
    with tab2:
        st.header("Simulate a Real-Time Transaction Feed")
        st.write(f"This will stream sample data compatible with **{model_choice}**. Use the controls to configure.")
        col1, col2 = st.columns(2)
        with col1:
            speed_options = {"Slow (2s)": 2.0, "Medium (1s)": 1.0, "Fast (0.5s)": 0.5, "Ludicrous! (0.1s)": 0.1}
            selected_speed_label = st.select_slider("Select Simulation Speed:", options=speed_options.keys(), value="Fast (0.5s)")
            simulation_delay = speed_options[selected_speed_label]
        with col2:
            dynamic_label = f"Transactions to simulate (1 - {max_rows:,}):"
            num_transactions = st.number_input(dynamic_label, min_value=1, max_value=max_rows, value=100, step=100)
        
        if st.button("🚀 Start Real-Time Simulation"):
            st.session_state.alerts = [] 
            simulation_df = sample_df.head(num_transactions)
            live_feed_placeholder = st.empty()
            live_alerts_placeholder = st.empty()
            
            for i in range(len(simulation_df)):
                transaction_row = simulation_df.iloc[[i]]
                with live_feed_placeholder.container():
                    st.info(f"▶️ Processing Transaction #{i+1}/{len(simulation_df)}...")
                    st.dataframe(transaction_row)

                processed_row = preprocessing_function(transaction_row, for_performance=False)
                prediction = model.predict(processed_row)[0]
                
                if prediction == 1:
                    raw_shap_values = explainer.shap_values(processed_row)
                    shap_values_to_plot = raw_shap_values[1][0] if isinstance(raw_shap_values, list) else raw_shap_values[0]
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                    shap_explanation = shap.Explanation(
                        values=shap_values_to_plot, 
                        base_values=expected_value, 
                        data=processed_row.iloc[0], 
                        feature_names=processed_row.columns.tolist()
                    )
                    st.session_state.alerts.append((transaction_row, shap_explanation))

                # Displays live alerts
                with live_alerts_placeholder.container():
                    st.subheader("🔴 Live Alert Log")
                    if not st.session_state.alerts:
                        st.info("No fraudulent transactions detected so far.")
                    else:
                        for alert_num, (alert_row, _) in enumerate(reversed(st.session_state.alerts)):
                            st.warning(f"**Alert #{len(st.session_state.alerts) - alert_num}: Potential Fraud Detected!**")
                            st.dataframe(alert_row)
                            with st.expander(f"🔍 Explanation for Alert #{len(st.session_state.alerts) - alert_num}"):
                                st.write("Detailed plot will be generated after the simulation completes.")
                
                time.sleep(simulation_delay)
                
            live_feed_placeholder.empty()

            with live_alerts_placeholder.container():
                st.subheader("🏁 Simulation Complete: Final Alert Log")
                if not st.session_state.alerts:
                    st.success("Scan complete. No fraudulent transactions were detected in the simulation.")
                else:
                    for alert_num, (alert_row, alert_shap) in enumerate(reversed(st.session_state.alerts)):
                        st.warning(f"**Alert #{len(st.session_state.alerts) - alert_num}: Potential Fraud Detected!**")
                        st.dataframe(alert_row)
                        with st.expander(f"🔍 See final explanation for Alert #{len(st.session_state.alerts) - alert_num}", expanded=True):
                            st_shap_plot(alert_shap)

            st.success("✅ Real-time simulation complete.")

    # Model Performance & Explainability Tab
    with tab3:
        st.header(f"Performance Metrics for: {model_choice}")
        st.write(f"This analysis is performed on the full sample dataset ('{str(sample_data_path).split('/')[-1]}') containing **{max_rows:,}** transactions.")

        if st.button("📈 Run Full Performance Analysis"):
            with st.spinner("Running predictions and calculating metrics..."):
                
                # Calls the preprocessing function with for_performance=True to get X and y
                X_perf, y_perf = preprocessing_function(sample_df, for_performance=True)

                # Makes predictions
                y_pred = model.predict(X_perf)
                y_scores = model.predict_proba(X_perf)[:, 1]

                # Calculates Metrics
                accuracy = accuracy_score(y_perf, y_pred)
                auprc = average_precision_score(y_perf, y_scores)
                roc_auc = roc_auc_score(y_perf, y_scores)
                precision = precision_score(y_perf, y_pred, zero_division=0)
                recall = recall_score(y_perf, y_pred, zero_division=0)
                f1 = f1_score(y_perf, y_pred, zero_division=0)
                report = classification_report(y_perf, y_pred, output_dict=True, zero_division=0)

            # Metrics Display
            st.subheader("📊 Overall Performance")
            
            # Primary metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Accuracy", f"{accuracy:.2%}")
            col2.metric("AUPRC", f"{auprc:.3f}", help="Area Under Precision-Recall Curve - Key metric for imbalanced datasets")
            col3.metric("ROC-AUC", f"{roc_auc:.3f}", help="Area Under ROC Curve")
            col4.metric("F1-Score (Fraud)", f"{f1:.3f}")
            
            # Secondary metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precision (Fraud)", f"{precision:.3f}", help="Of predicted frauds, how many were actually fraud?")
            col2.metric("Recall (Fraud)", f"{recall:.3f}", help="Of actual frauds, how many did were actually caught?")
            
            # Calculates business metrics
            total_fraud = (y_perf == 1).sum()
            detected_fraud = ((y_pred == 1) & (y_perf == 1)).sum()
            false_alarms = ((y_pred == 1) & (y_perf == 0)).sum()
            
            col3.metric("Frauds Detected", f"{detected_fraud}/{total_fraud}", help="Actual fraud cases caught by the model")
            col4.metric("False Alarms", f"{false_alarms:,}", help="Legitimate transactions flagged as fraud")

            # Add interpretation guide
            st.info("""
            🔍 **Metric Interpretation for Fraud Detection:**
            - **AUPRC**: Most important for imbalanced data. Higher is better (0-1 scale)
            - **Precision**: Minimises false alarms (flagging legitimate transactions)  
            - **Recall**: Maximises fraud detection (catching actual fraudulent transactions)
            - **F1-Score**: Balances precision and recall
            """)

            # Enhanced Visualisations
            st.subheader("📈 Visual Analysis")
            
            # Creates a 2x3 grid of visualisations
            fig = plt.figure(figsize=(18, 12), facecolor="#0E1117")
            
            # Confusion Matrix
            ax1 = plt.subplot(2, 3, 1)
            cm = confusion_matrix(y_perf, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                        xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
            ax1.set_title("Confusion Matrix", color='white', fontsize=12)
            ax1.set_ylabel('Actual Class', color='white')
            ax1.set_xlabel('Predicted Class', color='white')
            ax1.tick_params(colors='white')
            
            # Precision-Recall Curve
            ax2 = plt.subplot(2, 3, 2)
            pr_display = PrecisionRecallDisplay.from_predictions(y_perf, y_scores, ax=ax2, 
                                                               name=f'Model (AUPRC={auprc:.3f})')
            ax2.plot([0, 1], [total_fraud/len(y_perf), total_fraud/len(y_perf)], 
                    linestyle='--', label=f'No-Skill ({total_fraud/len(y_perf):.3f})')
            ax2.set_title("Precision-Recall Curve", color='white', fontsize=12)
            ax2.set_xlabel('Recall', color='white', fontsize=10)
            ax2.set_ylabel('Precision', color='white', fontsize=10)
            ax2.tick_params(colors='white')
            ax2.legend()
            
            # ROC Curve
            ax3 = plt.subplot(2, 3, 3)
            roc_display = RocCurveDisplay.from_predictions(y_perf, y_scores, ax=ax3, 
                                                         name=f'Model (AUC={roc_auc:.3f})')
            ax3.plot([0, 1], [0, 1], linestyle='--', label='No-Skill')
            ax3.set_title("ROC Curve", color='white', fontsize=12)
            ax3.set_xlabel('False Positive Rate', color='white', fontsize=10)
            ax3.set_ylabel('True Positive Rate', color='white', fontsize=10)
            ax3.tick_params(colors='white')
            ax3.legend()
            
            # Score Distribution
            ax4 = plt.subplot(2, 3, 4)
            fraud_scores = y_scores[y_perf == 1]
            normal_scores = y_scores[y_perf == 0]
            ax4.hist(normal_scores, bins=50, alpha=0.7, label='Not Fraud', density=True, color='blue')
            ax4.hist(fraud_scores, bins=50, alpha=0.7, label='Fraud', density=True, color='red')
            ax4.set_title("Prediction Score Distribution", color='white', fontsize=12)
            ax4.set_xlabel('Fraud Probability Score', color='white')
            ax4.set_ylabel('Density', color='white')
            ax4.tick_params(colors='white')
            ax4.legend()
            
            # Feature Importance
            ax5 = plt.subplot(2, 3, 5)
            # Gets feature importances. Handle ensemble vs. single model.
            if "Ensemble" in model_choice:
                # For ensemble, the RF component feature importances is used
                importances = model.named_estimators_['rf'].feature_importances_
            else: # For XGBoost
                importances = model.feature_importances_
            
            feature_names = X_perf.columns
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
            
            sns.barplot(x='importance', y='feature', data=importance_df, ax=ax5)
            ax5.set_title("Top 10 Most Important Features", color='white', fontsize=12)
            ax5.tick_params(colors='white')
            
            # Threshold Analysis
            ax6 = plt.subplot(2, 3, 6)
            precisions, recalls, thresholds = precision_recall_curve(y_perf, y_scores)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            
            # Calculates optimal threshold here so it can used in explanations
            optimal_threshold_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_threshold_idx]
            
            ax6.plot(thresholds, precisions[:-1], label='Precision', color='blue')
            ax6.plot(thresholds, recalls[:-1], label='Recall', color='red')
            ax6.plot(thresholds, f1_scores[:-1], label='F1-Score', color='green')
            ax6.axvline(x=optimal_threshold, color='yellow', linestyle=':', label=f'Optimal ({optimal_threshold:.3f})')
            ax6.set_title("Threshold Analysis", color='white', fontsize=12)
            ax6.set_xlabel('Classification Threshold', color='white')
            ax6.set_ylabel('Score', color='white')
            ax6.tick_params(colors='white')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explanations for each visualisation for better understanding
            st.subheader("📖 Visualisation Explanations")
            
            explanation_cols = st.columns(3)
            
            # Confusion Matrix Explanation
            with explanation_cols[0]:
                st.markdown("""
                **🔢 Confusion Matrix**
                
                Shows the breakdown of correct and incorrect predictions:
                - **True Negatives** (top-left): Legitimate transactions correctly identified
                - **False Positives** (top-right): False alarms - legitimate flagged as fraud
                - **False Negatives** (bottom-left): Missed fraud - fraud flagged as legitimate  
                - **True Positives** (bottom-right): Fraud correctly caught
                
                *Ideal: High numbers on the diagonal, low numbers off-diagonal*
                """)
                
            # Precision-Recall and ROC Curve Explanations    
            with explanation_cols[1]:
                st.markdown(f"""
                **📈 Precision-Recall Curve**
                
                Shows the trade-off between precision and recall:
                - **Higher curve** = better model performance
                - **AUPRC = {auprc:.3f}**: Area under this curve (closer to 1.0 is better)
                - **Baseline ({total_fraud/len(y_perf):.3f})**: Random guessing performance
                
                *This curve is the most important for imbalanced fraud detection*
                """)
                
            with explanation_cols[2]:
                st.markdown(f"""
                **🎯 ROC Curve**
                
                Shows true positive rate vs false positive rate:
                - **Closer to top-left corner** = better performance
                - **AUC = {roc_auc:.3f}**: Area under curve (0.5 = random, 1.0 = perfect)
                - **Diagonal line**: Random guessing
                
                *Good for comparing models, but less informative for imbalanced data*
                """)
            
            explanation_cols2 = st.columns(3)
            
            # Score Distribution, Feature Importance, Threshold Analysis Explanations
            with explanation_cols2[0]:
                st.markdown("""
                **📊 Score Distribution**
                
                Shows how the model's fraud probability scores are distributed:
                - **Blue (Not Fraud)**: Should be concentrated near 0
                - **Red (Fraud)**: Should be concentrated near 1
                - **Good separation** = distinct peaks with minimal overlap
                
                *Better separation means the model can distinguish fraud more clearly*
                """)
                
            with explanation_cols2[1]:
                st.markdown("""
                **🎯 Feature Importance**
                
                Shows which features the model relies on most:
                - **Longer bars** = more important for predictions
                - **Top features** drive most of the model's decisions
                - **Domain knowledge** should validate if important features make sense
                
                *Helps understand what patterns the model has learned*
                """)
                
            with explanation_cols2[2]:
                st.markdown(f"""
                **⚖️ Threshold Analysis**
                
                Shows how different classification thresholds affect performance:
                - **Precision (Blue)**: Decreases as threshold lowers (more false alarms)
                - **Recall (Red)**: Increases as threshold lowers (catch more fraud)
                - **F1-Score (Green)**: Balances both metrics
                
                *Current threshold: 0.5 | Recommended: {optimal_threshold:.3f}*
                """)
            
            st.markdown("---")
            
            # Model-Specific Insights
            st.subheader("🎯 Business Impact Analysis")
            
            # Calculates business metrics
            fraud_detection_rate = detected_fraud / total_fraud if total_fraud > 0 else 0
            false_alarm_rate = false_alarms / (len(y_perf) - total_fraud) if len(y_perf) > total_fraud else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Fraud Detection Rate", f"{fraud_detection_rate:.1%}", 
                         help="Percentage of actual frauds caught by the model")
                st.metric("False Alarm Rate", f"{false_alarm_rate:.3%}", 
                         help="Percentage of legitimate transactions incorrectly flagged")
                
            with col2:
                # Assuming average fraud amount vs legitimate transaction amount
                if "Credit Card" in model_choice:
                    avg_fraud_amount = 122.21  # From credit card dataset analysis
                    avg_legit_amount = 88.35
                else:  # Transactions
                    avg_fraud_amount = 1000000  # Typical for PaySim
                    avg_legit_amount = 100000
                    
                potential_fraud_caught = detected_fraud * avg_fraud_amount
                investigation_cost = false_alarms * 50  # Assumes $50 per false alarm investigation
                
                st.metric("Potential Fraud Value Caught", f"${potential_fraud_caught:,.2f}")
                st.metric("Investigation Cost (Est.)", f"${investigation_cost:,.2f}")
            
            # Model comparison insight
            if "Ensemble" in model_choice:
                st.success("🎯 **Ensemble Model Benefits**: Combines Random Forest's ability to capture complex patterns with Logistic Regression's interpretability and speed.")
            else:
                st.success("🚀 **XGBoost Benefits**: Excellent performance on structured data with built-in regularisation and efficient handling of missing values.")
                
            st.subheader("💡 Model Recommendations")
            
            if precision > 0.8 and recall > 0.7:
                st.success("✅ **Excellent Performance**: High precision and recall - model is production-ready!")
            elif precision > 0.6 and recall > 0.8:
                st.warning("⚠️ **High Recall, Moderate Precision**: Good at catching fraud but may have false alarms. Consider adjusting threshold.")
            elif precision > 0.8 and recall < 0.6:
                st.warning("⚠️ **High Precision, Low Recall**: Conservative model - misses some fraud but has few false alarms.")
            else:
                st.error("❌ **Needs Improvement**: Consider retraining with different parameters or more data.")
                
            st.info(f"💡 **Recommended Classification Threshold**: {optimal_threshold:.3f} (optimises F1-score)")

else:  # Model Comparison Mode
    st.sidebar.write("**Model Comparison Mode Selected**")
    
    # Dataset selection for comparison
    dataset_choice = st.sidebar.selectbox(
        "Choose Dataset for Comparison:",
        ("Credit Card Dataset", "Transactions Dataset")
    )
    
    st.sidebar.info(f"Comparing Ensemble vs XGBoost models on {dataset_choice}")
    
    # MODEL COMPARISON TAB
    st.header(f"🆚 Model Comparison: {dataset_choice}")
    st.write(f"Comprehensive comparison between Ensemble and XGBoost models on the **{dataset_choice}**")
    
    # Loads all models and data
    try:
        all_models, all_explainers = load_all_models()
        
        if dataset_choice == "Credit Card Dataset":
            sample_data_path = Path(__file__).parent.parent / "data" / "creditcard.csv"
            preprocessing_function = preprocess_credit_card_data
            model_names = ['Credit Card - Ensemble', 'Credit Card - XGBoost']
        else:  # Transactions Dataset
            sample_data_path = Path(__file__).parent.parent / "data" / "transactions_sample_280k.csv"
            preprocessing_function = preprocess_transactions_data
            model_names = ['Transactions - Ensemble', 'Transactions - XGBoost']
        
        sample_df = load_sample_data(sample_data_path)
        
    except FileNotFoundError as e:
        st.error(f"Error loading models or data: {e}")
        st.stop()
    
    if st.button("🚀 Run Model Comparison Analysis", type="primary"):
        with st.spinner("Running comprehensive model comparison..."):
            # Preprocesses data
            X_comp, y_comp = preprocessing_function(sample_df, for_performance=True)
            
            # Calculates metrics for both models
            metrics_comparison = {}
            for model_name in model_names:
                model = all_models[model_name]
                metrics_comparison[model_name.split(' - ')[1]] = calculate_model_metrics(model, X_comp, y_comp)
        
        st.success("✅ Comparison Analysis Complete!")
        
        # Performance Metrics Comparison
        st.subheader("📊 Performance Metrics Comparison")
        
        # Creates metrics comparison table
        metrics_df = pd.DataFrame({
            metric: [metrics_comparison[model][metric] for model in ['Ensemble', 'XGBoost']]
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'auprc']
        }, index=['Ensemble', 'XGBoost'])
        
        # Display metrics table with highlighting
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color='lightgreen')
                           .format("{:.4f}"),
            use_container_width=True
        )
        
        # Interactive Visualisations
        st.subheader("📈 Interactive Performance Visualisations")
        
        # Metrics comparison chart
        metrics_fig = create_comparison_visualisations(metrics_comparison, dataset_choice)
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # ROC and PR curves comparison
        curves_fig = create_roc_pr_comparison(metrics_comparison, X_comp, y_comp, dataset_choice)
        st.plotly_chart(curves_fig, use_container_width=True)
        
        # Detailed Analysis
        st.subheader("🔍 Detailed Model Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚡ **Ensemble Model**")
            ensemble_metrics = metrics_comparison['Ensemble']
            st.metric("Best Metric", "AUPRC", f"{ensemble_metrics['auprc']:.4f}")
            st.write("**Strengths:**")
            st.write("- Combines RF and LR for robust predictions")
            st.write("- Reduces overfitting through model averaging")
            st.write("- Generally more stable performance")
            
        with col2:
            st.markdown("### ⚡ **XGBoost Model**")
            xgb_metrics = metrics_comparison['XGBoost']
            st.metric("Best Metric", "F1-Score", f"{xgb_metrics['f1_score']:.4f}")
            st.write("**Strengths:**")
            st.write("- Excellent gradient boosting performance")
            st.write("- Built-in regularisation")
            st.write("- Efficient handling of imbalanced data")
        
        # Model Recommendation
        st.subheader("💡 Model Recommendation")
        
        # Determines which model performs better overall
        ensemble_score = (ensemble_metrics['auprc'] + ensemble_metrics['f1_score'] + ensemble_metrics['roc_auc']) / 3
        xgb_score = (xgb_metrics['auprc'] + xgb_metrics['f1_score'] + xgb_metrics['roc_auc']) / 3
        
        if ensemble_score > xgb_score:
            winner = "Ensemble"
            winner_color = "success"
        else:
            winner = "XGBoost"
            winner_color = "info"
        
        getattr(st, winner_color)(f"🎯 **Recommended Model for {dataset_choice}: {winner}**")
        
        # Detailed recommendations
        st.write("### 📋 Detailed Recommendations:")
        
        if ensemble_score > xgb_score:
            st.write("""
            - **Use Ensemble Model** for production deployment
            - Better overall stability and robustness
            - Lower risk of overfitting on new data
            - Suitable for scenarios requiring consistent performance
            """)
        else:
            st.write("""
            - **Use XGBoost Model** for production deployment
            - Superior performance on this specific dataset
            - More efficient for large-scale predictions
            - Better feature importance interpretability
            """)
        
        # Business Impact Analysis
        st.subheader("💼 Business Impact Analysis")
        
        # Calculates business metrics for both models
        total_fraud = (y_comp == 1).sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Ensemble Model Impact")
            ensemble_detected = ((metrics_comparison['Ensemble']['predictions'] == 1) & (y_comp == 1)).sum()
            ensemble_false_alarms = ((metrics_comparison['Ensemble']['predictions'] == 1) & (y_comp == 0)).sum()
            
            st.metric("Fraud Detection Rate", f"{ensemble_detected/total_fraud:.1%}")
            st.metric("Frauds Captured", f"{ensemble_detected}/{total_fraud}", help="Actual fraud cases caught by the model")
            st.metric("False Alarm Count", f"{ensemble_false_alarms:,}")
            
        with col2:
            st.markdown("#### XGBoost Model Impact")
            xgb_detected = ((metrics_comparison['XGBoost']['predictions'] == 1) & (y_comp == 1)).sum()
            xgb_false_alarms = ((metrics_comparison['XGBoost']['predictions'] == 1) & (y_comp == 0)).sum()
            
            st.metric("Fraud Detection Rate", f"{xgb_detected/total_fraud:.1%}")
            st.metric("Frauds Captured", f"{xgb_detected}/{total_fraud}", help="Actual fraud cases caught by the model")
            st.metric("False Alarm Count", f"{xgb_false_alarms:,}")
        
        # Feature Importance Comparison
        st.subheader("🎯 Feature Importance Comparison")
        
        # Get feature importances for both models
        if dataset_choice == "Credit Card Dataset":
            # For ensemble, uses RF component
            ensemble_importances = all_models['Credit Card - Ensemble'].named_estimators_['rf'].feature_importances_
            xgb_importances = all_models['Credit Card - XGBoost'].feature_importances_
        else:
            ensemble_importances = all_models['Transactions - Ensemble'].named_estimators_['rf'].feature_importances_
            xgb_importances = all_models['Transactions - XGBoost'].feature_importances_
        
        # Creates feature importance comparison
        feature_names = X_comp.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Ensemble': ensemble_importances,
            'XGBoost': xgb_importances
        }).sort_values('Ensemble', ascending=False).head(15)
        
        # Creates interactive feature importance plot
        fig_importance = go.Figure()
        
        fig_importance.add_trace(go.Bar(
            name='Ensemble',
            x=importance_df['Feature'],
            y=importance_df['Ensemble'],
            marker_color='#1f77b4'
        ))
        
        fig_importance.add_trace(go.Bar(
            name='XGBoost',
            x=importance_df['Feature'],
            y=importance_df['XGBoost'],
            marker_color='#ff7f0e'
        ))
        
        fig_importance.update_layout(
            title='Top 15 Feature Importance Comparison',
            xaxis_title='Features',
            yaxis_title='Importance',
            barmode='group',
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Confusion Matrix Comparison
        st.subheader("🔢 Confusion Matrix Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Ensemble Model")
            cm_ensemble = confusion_matrix(y_comp, metrics_comparison['Ensemble']['predictions'])
            
            fig_cm1, ax1 = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
            sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
            ax1.set_title("Ensemble Confusion Matrix", color='white')
            ax1.tick_params(colors='white')
            st.pyplot(fig_cm1)
            
        with col2:
            st.markdown("#### XGBoost Model")
            cm_xgb = confusion_matrix(y_comp, metrics_comparison['XGBoost']['predictions'])
            
            fig_cm2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
            sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                       xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
            ax2.set_title("XGBoost Confusion Matrix", color='white')
            ax2.tick_params(colors='white')
            st.pyplot(fig_cm2)
        
        # Threshold Analysis Comparison
        st.subheader("⚖️ Threshold Analysis Comparison")
        
        # Calculates optimal thresholds for both models
        precision_ens, recall_ens, thresholds_ens = precision_recall_curve(y_comp, metrics_comparison['Ensemble']['probabilities'])
        precision_xgb, recall_xgb, thresholds_xgb = precision_recall_curve(y_comp, metrics_comparison['XGBoost']['probabilities'])
        
        f1_scores_ens = 2 * (precision_ens * recall_ens) / (precision_ens + recall_ens + 1e-8)
        f1_scores_xgb = 2 * (precision_xgb * recall_xgb) / (precision_xgb + recall_xgb + 1e-8)
        
        optimal_threshold_ens = thresholds_ens[np.argmax(f1_scores_ens[:-1])]
        optimal_threshold_xgb = thresholds_xgb[np.argmax(f1_scores_xgb[:-1])]
        
        # Creates threshold comparison plot
        fig_threshold = go.Figure()
        
        fig_threshold.add_trace(go.Scatter(
            x=thresholds_ens,
            y=f1_scores_ens[:-1],
            mode='lines',
            name='Ensemble F1-Score',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_threshold.add_trace(go.Scatter(
            x=thresholds_xgb,
            y=f1_scores_xgb[:-1],
            mode='lines',
            name='XGBoost F1-Score',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        # Adds optimal threshold lines
        fig_threshold.add_vline(x=optimal_threshold_ens, line_dash="dash", 
                               line_color="#1f77b4", 
                               annotation_text=f"Ensemble Optimal: {optimal_threshold_ens:.3f}")
        fig_threshold.add_vline(x=optimal_threshold_xgb, line_dash="dash", 
                               line_color="#ff7f0e",
                               annotation_text=f"XGBoost Optimal: {optimal_threshold_xgb:.3f}")
        
        fig_threshold.update_layout(
            title='F1-Score vs Classification Threshold',
            xaxis_title='Threshold',
            yaxis_title='F1-Score',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_threshold, use_container_width=True)
        
        # Summary and Insights
        st.subheader("📝 Key Insights & Summary")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("### 🔍 **Performance Insights**")
            
            # Finds best performing metrics
            best_metrics = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'auprc']:
                if ensemble_metrics[metric] > xgb_metrics[metric]:
                    best_metrics[metric] = 'Ensemble'
                else:
                    best_metrics[metric] = 'XGBoost'
            
            for metric, winner in best_metrics.items():
                st.write(f"- **{metric.replace('_', ' ').title()}**: {winner} performs better")
        
        with insights_col2:
            st.markdown("### 💡 **Recommendations**")
            
            if dataset_choice == "Credit Card Dataset":
                st.write("""
                **For Credit Card Fraud:**
                - High precision crucial (minimise false alarms)
                - Real-time processing requirements
                - Regulatory compliance considerations
                """)
            else:
                st.write("""
                **For Transaction Fraud:**
                - Balance between detection and false alarms
                - Handle high transaction volumes
                - Consider transaction type patterns
                """)
        
        
        # Model Deployment Recommendations
        st.subheader("🚀 Deployment Recommendations")
        
        deployment_col1, deployment_col2 = st.columns(2)
        
        with deployment_col1:
            st.markdown("### 🎯 **Production Deployment**")
            
            if winner == "Ensemble":
                st.success(f"""
                **Deploy Ensemble Model** for {dataset_choice}
                
                ✅ **Advantages:**
                - More robust and stable predictions
                - Lower variance in performance
                - Better generalisation to new data
                - Reduced overfitting risk
                """)
            else:
                st.info(f"""
                **Deploy XGBoost Model** for {dataset_choice}
                
                ✅ **Advantages:**
                - Superior performance metrics
                - Faster inference time
                - Better feature interpretability
                - More efficient resource usage
                """)
        
        with deployment_col2:
            st.markdown("### ⚙️ **Implementation Considerations**")
            
            st.write(f"""
            **Optimal Threshold:**
            - Ensemble: {optimal_threshold_ens:.3f}
            - XGBoost: {optimal_threshold_xgb:.3f}
            
            **Monitoring Metrics:**
            - Track F1-score and AUPRC
            - Monitor false alarm rates
            - Review feature importance drift
            
            **A/B Testing:**
            - Consider running both models in parallel
            - Compare real-world performance
            - Gradual rollout recommended
            """)