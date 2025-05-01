# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, accuracy_score
)
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="Machinery Efficiency Evaluation",
    layout="wide",
)

# --- Suppress Warnings (Optional) ---
warnings.filterwarnings("ignore", message="use_label_encoder is deprecated")
# Filter out FutureWarning from seaborn countplot/boxplot regarding palette
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn._oldcore")


# --- Data Generation ---
@st.cache_data
def generate_data(num_samples=2500):
    """Generates synthetic machinery data based on the new Colab code."""
    np.random.seed(42)
    N = num_samples
    Shift = np.random.choice(['Morning', 'Evening', 'Night'], N)
    Device = np.random.choice(['A', 'B', 'C', 'D'], N)
    fail_prob = {'A': 0.05, 'B': 0.10, 'C': 0.20, 'D': 0.15}
    Failure = np.array([np.random.rand() < fail_prob[d] for d in Device])
    base_by_shift = {'Morning': 50, 'Evening': 30, 'Night': 20}
    dev_modifier = {'A': 0, 'B': 5, 'C': -5, 'D': 2}
    NoOrder = []
    for i in range(N):
        base = base_by_shift[Shift[i]] + dev_modifier[Device[i]]
        if Failure[i]:
            base -= 30
        val = base + np.random.normal(0, 5)
        # Corrected Line Below (removed extra parenthesis):
        NoOrder.append(max(0, int(round(val)))) # Ensure integer and >= 0
    NoOrder = np.array(NoOrder)

    shift_eff = {'Morning': 5, 'Evening': -5, 'Night': 10}
    device_eff = {'A': 0, 'B': -2, 'C': -5, 'D': 3}
    Efficiency = []
    for i in range(N):
        eff = 70
        eff += shift_eff[Shift[i]]
        eff += device_eff[Device[i]]
        if Failure[i]:
            eff -= 15
        no = NoOrder[i]
        eff += 15 * np.exp(-((no - 30) ** 2) / (2 * 30 ** 2))
        eff -= 0.1 * no
        if Shift[i] == 'Night' and Device[i] == 'C':
            eff += 10
        eff += np.random.normal(0, 5)
        Efficiency.append(min(max(eff, 0), 100))
    Efficiency = np.array(Efficiency)
    Efficient = (Efficiency >= 90).astype(int)

    df = pd.DataFrame({
        'Shift': Shift,
        'Device': Device,
        'Failure': Failure,
        'NoOrder': NoOrder,
        'Efficiency': np.round(Efficiency, 2), # Round for better display
        'Efficient': Efficient
    })
    df['Failure'] = df['Failure'].map({True: 'yes', False: 'no'})
    return df

# --- Data Preprocessing ---
@st.cache_data # Cache preprocessing results
def preprocess_data(df):
    """Encodes categorical features, scales data, and splits into train/test."""
    df_encoded = df.copy()
    label_encoders = {}
    # Ensure Failure is string before encoding
    df_encoded['Failure'] = df_encoded['Failure'].astype(str)

    categorical_cols = ['Shift', 'Device', 'Failure']
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Prepare df for correlation visualization (only encode necessary categoricals)
    df_corr_vis = df.copy()
    df_corr_vis['Failure'] = LabelEncoder().fit_transform(df_corr_vis['Failure'])
    df_corr_vis['Shift'] = LabelEncoder().fit_transform(df_corr_vis['Shift'])
    df_corr_vis['Device'] = LabelEncoder().fit_transform(df_corr_vis['Device'])


    feature_cols = ['Shift', 'Device', 'Failure', 'NoOrder']
    X = df_encoded[feature_cols]
    y = df_encoded['Efficient']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols, df_corr_vis, scaler, label_encoders # Return scaler & encoders if needed

# --- Visualization Functions ---
def plot_eda_countplot_page(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Efficient', palette='Set2', ax=ax)
    ax.set_title('Efficiency Class Distribution')
    ax.set_xlabel('Efficient (1 = Yes, 0 = No)')
    ax.set_ylabel('Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Efficient', 'Efficient'])
    return fig

def plot_eda_boxplot_shift_page(df):
    Q1 = df['Efficiency'].quantile(0.25)
    Q3 = df['Efficiency'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df['Efficiency'] >= lower_bound) & (df['Efficiency'] <= upper_bound)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_clean, x='Shift', y='Efficiency', palette='Set2', ax=ax)
    ax.set_title('Efficiency by Shift (Outliers Removed)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_xlabel('Shift')
    return fig

def plot_eda_boxplot_device_page(df):
    Q1 = df['Efficiency'].quantile(0.25)
    Q3 = df['Efficiency'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df['Efficiency'] >= lower_bound) & (df['Efficiency'] <= upper_bound)]
    fig, ax = plt.subplots(figsize=(10, 5)) # Adjusted size
    sns.boxplot(data=df_filtered, x='Device', y='Efficiency', palette='coolwarm', ax=ax)
    ax.set_title('Efficiency by Device (Outliers Removed)')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Efficiency (%)')
    return fig

def plot_eda_failure_noorder_counts_page(df):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x='Failure', ax=axs[0], palette='Set1')
    axs[0].set_title("Failure Distribution")
    axs[0].set_xlabel("Failure (yes/no)")
    sns.histplot(data=df, x='NoOrder', ax=axs[1], kde=True, bins=20) # Use default palette
    axs[1].set_title("No Order Count Distribution")
    axs[1].set_xlabel("No Order Count")
    return fig

def plot_eda_correlation_heatmap_page(df_corr_vis):
    # Calculate correlation on the specifically prepared DataFrame
    corr = df_corr_vis[['Shift', 'Device', 'Failure', 'NoOrder', 'Efficiency', 'Efficient']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax) # format numbers
    ax.set_title("Feature Correlation Heatmap")
    return fig

def plot_eda_qq_plots_page(df):
    st.subheader("Normal Q-Q Plots of Efficiency by Shift")
    shifts = sorted(df['Shift'].unique())
    cols = st.columns(len(shifts))
    for i, shift in enumerate(shifts):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(5, 4))
            stats.probplot(df[df['Shift'] == shift]['Efficiency'], dist="norm", plot=ax) # Plot on ax
            ax.set_title(f"Shift: {shift}")
            ax.grid(True)
            ax.set_xlabel("") # Keep titles clean in columns
            ax.set_ylabel("")
            st.pyplot(fig) # Display directly within the column

def plot_probability_scatter(y_test, y_proba, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    y_test_array = y_test.reset_index(drop=True) # Ensure index alignment
    scatter = ax.scatter(range(len(y_proba)), y_proba, c=y_test_array, cmap='coolwarm', alpha=0.6)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.set_title(title)
    ax.set_ylabel('Predicted Probability of Being Efficient')
    ax.set_xlabel('Sample Index (Test Set)')
    ax.grid(True)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(0.), markersize=8, label='Not Efficient'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(1.), markersize=8, label='Efficient')]
    ax.legend(handles=handles, title="Actual Class")
    return fig

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Efficient', 'Efficient'],
                yticklabels=['Not Efficient', 'Efficient'])
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def plot_roc_curve(y_test, y_proba, title="Receiver Operating Characteristic Curve"):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig, roc_auc

# --- Model Training and Evaluation (Cached) ---
@st.cache_data(show_spinner=False) # Cache results for each model type
def train_and_evaluate(model_name, X_train, X_test, y_train, y_test):
    """Trains the selected model and returns evaluation results."""
    if model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        st.error("Invalid model selected.")
        return None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": model_name,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    return results

# --- ROC Comparison Function ---
def plot_roc_comparison(y_test, all_probas):
    """Plots ROC curves for all models on the same graph."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'Logistic Regression': 'blue', 'Random Forest': 'green', 'SVM': 'red', 'XGBoost': 'purple'}

    # Corrected Loop Below: Iterate directly over the probabilities passed
    for name, y_proba in all_probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba) # Use y_proba directly
        roc_auc = roc_auc_score(y_test, y_proba) # Calculate AUC here
        ax.plot(fpr, tpr, color=colors.get(name, 'black'), label=f'{name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    ax.grid(True)
    return fig

# --- Main Application Logic ---

st.title("Production Machinery Efficiency Evaluation")
st.markdown("BATCH: 3 | GROUP: 15 | DV-CP | ROLL NO: 4, 6, 12, 15")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
# Removed emojis, added numbers
page = st.sidebar.radio("Select Section:",
                       ('1. Home',
                        '2. Dataset Insights',
                        '3. Logistic Regression',
                        '4. Random Forest',
                        '5. SVM',
                        '6. XGBoost',
                        '7. Comparison'))

# --- Load and Preprocess Data (Run once, cached) ---
df_original = generate_data()
X_train, X_test, y_train, y_test, feature_cols, df_corr_vis, scaler, label_encoders = preprocess_data(df_original)


# --- Page Content ---
if page == '1. Home':
    st.header("Welcome!")
    st.markdown(
        """
        This application evaluates machinery efficiency using synthetic data and various
        classification algorithms.

        **Use the sidebar to navigate between sections:**

        *   **2. Dataset Insights:** Explore the data visually.
        *   **3. Logistic Regression, 4. Random Forest, 5. SVM, 6. XGBoost:** See individual model results.
        *   **7. Comparison:** Compare model performance using ROC curves.
        """
    )
    st.header("Data Overview")
    st.dataframe(df_original.head())
    st.write(f"*Dataset contains {len(df_original)} records.*")

elif page == '2. Dataset Insights':
    st.title("2. Dataset Insights & EDA") # Added number
    st.markdown("Exploring the characteristics of the generated machinery data.")
    st.dataframe(df_original.describe()) # Add descriptive stats

    # Arrange plots using columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Efficiency Target Distribution")
        st.pyplot(plot_eda_countplot_page(df_original))
        st.subheader("Failure & No Order Distributions")
        st.pyplot(plot_eda_failure_noorder_counts_page(df_original))
    with col2:
        st.subheader("Correlation Heatmap")
        st.pyplot(plot_eda_correlation_heatmap_page(df_corr_vis)) # Use df_corr_vis
        st.subheader("Efficiency by Shift")
        st.pyplot(plot_eda_boxplot_shift_page(df_original))

    st.subheader("Efficiency by Device")
    st.pyplot(plot_eda_boxplot_device_page(df_original))

    plot_eda_qq_plots_page(df_original) # This function handles its own subheader and plotting

elif page == '3. Logistic Regression':
    st.title("3. Logistic Regression Results") # Added number
    with st.spinner("Training Logistic Regression model..."):
        results = train_and_evaluate('Logistic Regression', X_train, X_test, y_train, y_test)

    if results:
        st.header("Evaluation Metrics")
        report_df = pd.DataFrame(results["report"]).transpose()
        st.dataframe(report_df.round(3))
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col_m2:
            st.metric("ROC AUC Score", f"{results['roc_auc']:.4f}")

        st.header("Visualizations")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.pyplot(plot_confusion_matrix(y_test, results["y_pred"], "Confusion Matrix - Logistic Regression"))
        with col_v2:
            fig_roc, _ = plot_roc_curve(y_test, results["y_proba"], "ROC Curve - Logistic Regression")
            st.pyplot(fig_roc)

        st.pyplot(plot_probability_scatter(y_test, results["y_proba"], "Predicted Probabilities - Logistic Regression"))
    else:
        st.error("Failed to train model.")

elif page == '4. Random Forest':
    st.title("4. Random Forest Results") # Added number
    with st.spinner("Training Random Forest model..."):
        results = train_and_evaluate('Random Forest', X_train, X_test, y_train, y_test)

    if results:
        st.header("Evaluation Metrics")
        report_df = pd.DataFrame(results["report"]).transpose()
        st.dataframe(report_df.round(3))
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col_m2:
            st.metric("ROC AUC Score", f"{results['roc_auc']:.4f}")

        st.header("Visualizations")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.pyplot(plot_confusion_matrix(y_test, results["y_pred"], "Confusion Matrix - Random Forest"))
        with col_v2:
            fig_roc, _ = plot_roc_curve(y_test, results["y_proba"], "ROC Curve - Random Forest")
            st.pyplot(fig_roc)
    else:
        st.error("Failed to train model.")


elif page == '5. SVM':
    st.title("5. Support Vector Machine (SVM) Results") # Added number
    with st.spinner("Training SVM model... This might take a moment."):
        results = train_and_evaluate('SVM', X_train, X_test, y_train, y_test)

    if results:
        st.header("Evaluation Metrics")
        report_df = pd.DataFrame(results["report"]).transpose()
        st.dataframe(report_df.round(3))
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col_m2:
            st.metric("ROC AUC Score", f"{results['roc_auc']:.4f}")

        st.header("Visualizations")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.pyplot(plot_confusion_matrix(y_test, results["y_pred"], "Confusion Matrix - SVM"))
        with col_v2:
            fig_roc, _ = plot_roc_curve(y_test, results["y_proba"], "ROC Curve - SVM")
            st.pyplot(fig_roc)
    else:
        st.error("Failed to train model.")

elif page == '6. XGBoost':
    st.title("6. XGBoost Results") # Added number
    with st.spinner("Training XGBoost model..."):
        results = train_and_evaluate('XGBoost', X_train, X_test, y_train, y_test)

    if results:
        st.header("Evaluation Metrics")
        report_df = pd.DataFrame(results["report"]).transpose()
        st.dataframe(report_df.round(3))
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col_m2:
            st.metric("ROC AUC Score", f"{results['roc_auc']:.4f}")

        st.header("Visualizations")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.pyplot(plot_confusion_matrix(y_test, results["y_pred"], "Confusion Matrix - XGBoost"))
        with col_v2:
            fig_roc, _ = plot_roc_curve(y_test, results["y_proba"], "ROC Curve - XGBoost")
            st.pyplot(fig_roc)
    else:
        st.error("Failed to train model.")

elif page == '7. Comparison':
    st.title("7. Model Comparison") # Added number
    st.markdown("Comparing the performance of all trained classification models.")

    st.header("Performance Summary & ROC Curves")
    # --- Get Results for All Models (uses cache) ---
    all_results = {}
    model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost']
    # Show spinner for the whole comparison process
    with st.spinner("Training all models for comparison... This might take a moment."):
        for name in model_names:
            results = train_and_evaluate(name, X_train, X_test, y_train, y_test)
            if results:
                all_results[name] = results
            else:
                st.warning(f"Could not get results for {name}")


    if all_results:
        # Display comparison table
        comparison_data = {
            "Algorithm": list(all_results.keys()),
            "Accuracy": [res["accuracy"] for res in all_results.values()],
            "ROC AUC": [res["roc_auc"] for res in all_results.values()]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.round(4).set_index("Algorithm"))

        # Plot comparison ROC
        # Corrected Data Structure for Plotting Function:
        probas_for_plot = {name: res["y_proba"] for name, res in all_results.items()}
        fig_comp = plot_roc_comparison(y_test, probas_for_plot)
        st.pyplot(fig_comp)
    else:
        st.error("Could not retrieve results for model comparison.")


# --- Footer or End Note ---
st.sidebar.info("End of Application Sections") # Optional