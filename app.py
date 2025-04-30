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

# --- Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="Machinery Efficiency Evaluation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Generation ---
@st.cache_data  # Cache the generated data
def generate_data(num_samples=10000):
    """Generates synthetic machinery data."""
    np.random.seed(42)
    shifts = np.random.choice(['1st', '2nd', '3rd'], size=num_samples, p=[0.3, 0.35, 0.35])
    device_list = ['H2', 'H4', 'H5', 'H6', 'H12', 'H14', 'H21', 'H22', 'H23', 'H24', 'H25']
    devices = np.random.choice(device_list, size=num_samples)
    failures = np.random.choice(['yes', 'no'], size=num_samples, p=[0.3, 0.7])
    no_orders = np.random.choice(['yes', 'no'], size=num_samples, p=[0.1, 0.9])

    efficiency = np.random.normal(loc=92, scale=5, size=num_samples)
    efficiency -= np.where(failures == 'yes', np.random.uniform(5, 10, num_samples), 0)
    efficiency -= np.where(no_orders == 'yes', np.random.uniform(10, 15, num_samples), 0)
    efficiency -= np.where(shifts == '1st', np.random.uniform(1, 3, num_samples), 0)
    efficiency = np.clip(efficiency, 40, 100)

    efficient_binary = np.where(efficiency >= 90, 1, 0)

    synthetic_df = pd.DataFrame({
        'Shift': shifts,
        'Device': devices,
        'Failure': failures,
        'NoOrder': no_orders,
        'Efficiency': np.round(efficiency, 2),
        'Efficient': efficient_binary
    })
    return synthetic_df

# --- Data Preprocessing ---
@st.cache_data # Cache preprocessing results
def preprocess_data(df):
    """Encodes categorical features, scales data, and splits into train/test."""
    df_encoded = df.copy()
    label_encoders = {}
    categorical_cols = ['Shift', 'Device', 'Failure', 'NoOrder']
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    feature_cols = ['Shift', 'Device', 'Failure', 'NoOrder'] # Keep original feature names if needed later
    X = df_encoded[feature_cols]
    y = df_encoded['Efficient']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols, df_encoded # Return encoded df for heatmap

# --- Visualization Functions ---

def plot_eda_countplot(df):
    """Plots the count of Efficient vs Not Efficient."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Efficient', palette='Set2', ax=ax)
    ax.set_title('Efficiency Class Distribution (Efficient vs Not)')
    ax.set_xlabel('Efficient (1 = Yes, 0 = No)')
    ax.set_ylabel('Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Efficient', 'Efficient'])
    st.pyplot(fig)

def plot_eda_boxplot_shift(df):
    """Plots boxplot of Efficiency by Shift after removing outliers."""
    Q1 = df['Efficiency'].quantile(0.25)
    Q3 = df['Efficiency'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df['Efficiency'] >= lower_bound) & (df['Efficiency'] <= upper_bound)]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_clean, x='Shift', y='Efficiency', palette='Set2', ax=ax)
    ax.set_title('Efficiency by Shift (Outliers Removed)')
    ax.set_ylabel('Efficiency')
    ax.set_xlabel('Shift')
    st.pyplot(fig)

def plot_eda_boxplot_device(df):
    """Plots boxplot of Efficiency by Device after removing outliers."""
    Q1 = df['Efficiency'].quantile(0.25)
    Q3 = df['Efficiency'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df['Efficiency'] >= lower_bound) & (df['Efficiency'] <= upper_bound)]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_filtered, x='Device', y='Efficiency', palette='coolwarm', ax=ax)
    ax.set_title('Efficiency by Device (Outliers Removed)')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_eda_failure_noorder_counts(df):
    """Plots countplots for Failure and NoOrder."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x='Failure', ax=axs[0], palette='Set1')
    axs[0].set_title("Failure Distribution")
    sns.countplot(data=df, x='NoOrder', ax=axs[1], palette='Set2')
    axs[1].set_title("No Order Distribution")
    st.pyplot(fig)

def plot_eda_correlation_heatmap(df_encoded):
    """Plots the correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap (Encoded Features)")
    st.pyplot(fig)

def plot_eda_qq_plots(df):
    """Plots Normal Q-Q plots for Efficiency grouped by Shift."""
    shifts = df['Shift'].unique()
    st.write("Normal Q-Q Plots of Efficiency by Shift:")
    cols = st.columns(len(shifts))
    for i, shift in enumerate(shifts):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(5, 4)) # Smaller plots for columns
            stats.probplot(df[df['Shift'] == shift]['Efficiency'], dist="norm", plot=plt)
            ax.set_title(f"Shift: {shift}")
            ax.grid(True)
            st.pyplot(fig)

def plot_probability_scatter(y_test, y_proba, title):
    """Plots the predicted probabilities scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(range(len(y_proba)), y_proba, c=y_test, cmap='coolwarm', alpha=0.6)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.set_title(title)
    ax.set_ylabel('Predicted Probability of Being Efficient')
    ax.set_xlabel('Sample Index (Test Set)')
    ax.grid(True)
    # Add legend - create dummy plots for handles
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(0.), markersize=8, label='Actual: Not Efficient'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(1.), markersize=8, label='Actual: Efficient')]
    ax.legend(handles=handles, title="Actual Class")
    st.pyplot(fig)

def plot_confusion_matrix(y_test, y_pred, title):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Efficient', 'Efficient'],
                yticklabels=['Not Efficient', 'Efficient'])
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

def plot_roc_curve(y_test, y_proba, title, model_name):
    """Plots the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- Model Training and Evaluation ---
# @st.cache_resource # Cache the trained model and results (optional)
def train_and_evaluate(model_name, X_train, X_test, y_train, y_test):
    """Trains the selected model and returns evaluation results."""
    if model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'SVM':
        model = SVC(probability=True, random_state=42) # probability needed for ROC AUC
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') # Suppress warning
    else:
        st.error("Invalid model selected.")
        return None, None, None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results = {
        "report": report,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "model": model # Return model if needed elsewhere
    }
    return results

# --- ROC Comparison Function ---
# @st.cache_data # Cache the results for the comparison plot
def get_all_model_probas(X_train, X_test, y_train, y_test):
    """Trains all models and returns their probabilities for ROC comparison."""
    probas = {}
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        probas[name] = model.predict_proba(X_test)[:, 1]
    return probas

def plot_roc_comparison(y_test, all_probas):
    """Plots ROC curves for all models on the same graph."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'Logistic Regression': 'blue', 'Random Forest': 'green', 'SVM': 'red', 'XGBoost': 'purple'}

    for name, y_proba in all_probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=colors[name], label=f'{name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# --- Main Application ---
st.title("Logistic Regression for Production Machinery Efficiency Evaluation")
st.markdown("BATCH: 3 | GROUP: 15 | DV-CP | ROLL NO: 4, 6, 12, 15")

# -- Data Loading & Display --
df_original = generate_data()
st.header("Generated Synthetic Data")
with st.expander("Show Data Sample"):
    st.dataframe(df_original.head())
st.write(f"Dataset created with {len(df_original)} rows.")
st.write("**Target Class Distribution (Efficient):**")
st.bar_chart(df_original['Efficient'].value_counts())


# -- Preprocessing --
X_train, X_test, y_train, y_test, feature_names, df_encoded = preprocess_data(df_original)

# -- EDA Visualizations --
st.header("Exploratory Data Analysis")

# Arrange EDA plots in columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Efficiency Distribution")
    plot_eda_countplot(df_original)
    st.subheader("Failure & No Order Counts")
    plot_eda_failure_noorder_counts(df_original)


with col2:
    st.subheader("Correlation Heatmap")
    plot_eda_correlation_heatmap(df_encoded) # Use encoded df for heatmap
    st.subheader("Efficiency by Shift")
    plot_eda_boxplot_shift(df_original)


st.subheader("Efficiency by Device")
plot_eda_boxplot_device(df_original)

st.subheader("Normality Plots")
plot_eda_qq_plots(df_original)


# -- Model Selection and Results --
st.sidebar.title("Algorithm Selection")
algorithm_list = ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost']
selected_algo = st.sidebar.selectbox("Choose a classification algorithm:", algorithm_list)

st.header(f"Machine Learning Model: {selected_algo}")

# Train and evaluate the selected model
results = train_and_evaluate(selected_algo, X_train, X_test, y_train, y_test)

if results:
    st.subheader("Evaluation Metrics")
    st.text("Classification Report:")
    st.text(results["report"])
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Accuracy", f"{results['accuracy']:.4f}")
    with col_m2:
        st.metric("ROC AUC Score", f"{results['roc_auc']:.4f}")


    st.subheader("Visualizations for Selected Model")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        plot_confusion_matrix(y_test, results["y_pred"], f"Confusion Matrix - {selected_algo}")
    with col_v2:
        plot_roc_curve(y_test, results["y_proba"], f"ROC Curve - {selected_algo}", selected_algo)

    # Specific plot for Logistic Regression
    if selected_algo == 'Logistic Regression':
         st.subheader("Predicted Probability Scatter Plot")
         plot_probability_scatter(y_test, results["y_proba"], f"Predicted Probabilities - {selected_algo}")


# -- ROC Curve Comparison --
st.header("Model Comparison")
st.subheader("ROC Curve Comparison (All Models)")
with st.spinner("Training all models for comparison..."):
    all_probas = get_all_model_probas(X_train, X_test, y_train, y_test)
    plot_roc_comparison(y_test, all_probas)

st.success("App finished!")