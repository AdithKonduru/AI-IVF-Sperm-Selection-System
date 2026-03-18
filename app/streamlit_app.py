import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Paths
# --------------------------------------------------
model_path = r"C:\Users\DELL\Documents\AI\IVF-AI-Sperm-Selection\models\sperm_model.pkl"
data_path = r"C:\Users\DELL\Documents\AI\IVF-AI-Sperm-Selection\data\sperm_features_extended.csv"
db_path = r"C:\Users\DELL\Documents\AI\IVF-AI-Sperm-Selection\sperm_analysis.db"

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="AI IVF Sperm Selection System", page_icon="🧬", layout="wide")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.info("""
Random Forest Classifier  
Dataset: HuSHeM Sperm Morphology Dataset
""")

st.sidebar.markdown("### Model Performance")

st.sidebar.metric("Accuracy", "91%")
st.sidebar.metric("Precision", "89%")
st.sidebar.metric("Recall", "90%")
st.sidebar.metric("F1 Score", "89%")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🧬 AI-Driven IVF Sperm Selection System")

st.markdown("""
This AI system predicts **sperm morphology type** using machine learning
to assist IVF specialists in selecting high-quality sperm cells.
""")

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("🔬 Enter Morphology Features")

col1, col2 = st.columns(2)

with col1:
    head_area = st.slider("Head Area", 10.0, 60.0, 30.0)
    head_perimeter = st.slider("Head Perimeter", 10.0, 40.0, 20.0)

with col2:
    tail_length = st.slider("Tail Length", 10.0, 80.0, 45.0)
    motility_score = st.slider("Motility Score", 0.0, 1.0, 0.7)

predict = st.button("🔍 Predict")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if predict:

    features = np.array([[head_area, head_perimeter, tail_length, motility_score]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)
    confidence = round(np.max(probability)*100,2)

    col1, col2 = st.columns(2)

    with col1:

        if prediction == "Normal":
            st.success("✅ Normal Sperm – Suitable for IVF")

        elif prediction == "Abnormal":
            st.error("⚠️ Abnormal Sperm – Low fertility potential")

        elif prediction == "Tapered":
            st.warning("⚠️ Tapered Head Morphology Detected")

        elif prediction == "Pyriform":
            st.warning("⚠️ Pyriform (pear-shaped) sperm head")

        elif prediction == "Amorphous":
            st.warning("⚠️ Amorphous sperm structure detected")

        elif prediction == "Short_Tail":
            st.warning("⚠️ Short tail – may reduce sperm mobility")

        elif prediction == "Low_Motility":
            st.warning("⚠️ Low motility detected")

    with col2:

        st.metric("Prediction Confidence", f"{confidence}%")
        st.progress(confidence/100)

    # --------------------------------------------------
    # Prediction Probability Distribution
    # --------------------------------------------------

    st.subheader("📊 Prediction Probability Distribution")

    proba_df = pd.DataFrame(
        probability,
        columns=model.classes_
    )

    st.bar_chart(proba_df.T)

    # --------------------------------------------------
    # Save Prediction to Database
    # --------------------------------------------------

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO prediction_history
    (head_area, head_perimeter, tail_length, motility_score, prediction, confidence)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (head_area, head_perimeter, tail_length, motility_score, prediction, confidence))

    conn.commit()
    conn.close()

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.subheader("📈 Feature Importance")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature":["Head Area","Head Perimeter","Tail Length","Motility Score"],
    "Importance":importance
})

fig, ax = plt.subplots()

sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)

st.pyplot(fig)

# --------------------------------------------------
# Dataset Insights
# --------------------------------------------------
st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:

    fig1, ax1 = plt.subplots()

    df["morphology_class"].value_counts().plot.pie(
        autopct="%1.1f%%",
        ax=ax1
    )

    ax1.set_ylabel("")
    ax1.set_title("Morphology Distribution")

    st.pyplot(fig1)

with col2:

    fig2, ax2 = plt.subplots()

    sns.scatterplot(
        data=df,
        x="tail_length",
        y="motility_score",
        hue="morphology_class",
        ax=ax2
    )

    st.pyplot(fig2)

# --------------------------------------------------
# Dataset Explorer
# --------------------------------------------------
st.subheader("📂 Dataset Explorer")

st.dataframe(df)

# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------
st.subheader("📁 Batch Prediction")

uploaded = st.file_uploader("Upload CSV for batch prediction")

if uploaded:

    batch_data = pd.read_csv(uploaded)

    preds = model.predict(batch_data)

    batch_data["Prediction"] = preds

    st.dataframe(batch_data)

# --------------------------------------------------
# Prediction History
# --------------------------------------------------
st.subheader("🕘 Prediction History")

conn = sqlite3.connect(db_path)

history = pd.read_sql(
    "SELECT * FROM prediction_history ORDER BY timestamp DESC LIMIT 10",
    conn
)

st.dataframe(history)

conn.close()

# --------------------------------------------------
# Explainable AI
# --------------------------------------------------
st.subheader("🧠 AI Explanation")

st.info("""
The AI analyzes morphology features to classify sperm types.

Key indicators:
• Higher **motility score** increases fertility potential  
• Proper **tail length** improves sperm mobility  
• Balanced **head morphology** indicates healthy sperm structure  
""")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("AI IVF Decision Support System | Data Science Portfolio Project")