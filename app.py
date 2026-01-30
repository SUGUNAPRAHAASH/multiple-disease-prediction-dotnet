"""
Multi-Disease Prediction System
A professional healthcare prediction application built with Streamlit.
SINGLE FILE VERSION - All code in one file for easy deployment.

Author: AI Healthcare Solutions
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# ============================================
# CUSTOM CSS STYLES
# ============================================

def apply_custom_styles():
    """Apply custom CSS styles for professional healthcare UI."""
    st.markdown("""
        <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Main container */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid #e2e8f0;
        }

        /* Cards */
        .health-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }

        /* Result cards */
        .result-positive {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 2px solid #fca5a5;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }

        .result-negative {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 2px solid #6ee7b7;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }

        /* Risk badges */
        .risk-low {
            background: #10b981;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 700;
        }

        .risk-medium {
            background: #f59e0b;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 700;
        }

        .risk-high {
            background: #ef4444;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 700;
        }

        /* Disclaimer */
        .disclaimer {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 1px solid #f59e0b;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }

        /* Headers */
        h1 { color: #1e3a5f !important; }
        h2 { color: #334155 !important; }
        h3 { color: #475569 !important; }
        </style>
    """, unsafe_allow_html=True)


def render_disclaimer():
    """Render the medical disclaimer."""
    st.markdown("""
        <div class="disclaimer">
            <p style="color: #92400e; font-size: 0.875rem; margin: 0;">
                <strong>⚠️ Medical Disclaimer:</strong> This application by MedIndia provides informational insights only
                and is not a medical diagnosis. Please consult a healthcare provider for proper diagnosis and treatment.
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_medindia_footer():
    """Render MedIndia footer on each page."""
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                    border-radius: 10px; margin-top: 1rem;">
            <img src="https://medindia.net/images/common/medindia-logo.png"
                 alt="MedIndia Logo"
                 style="max-width: 100px; height: auto; margin-bottom: 0.5rem;">
            <p style="color: #1e3a5f; font-size: 0.85rem; margin: 0;">
                <strong>MedIndia's HealthPredict AI</strong>
            </p>
            <p style="color: #3b82f6; font-size: 0.8rem; margin: 0;">
                Empowering Better Health
            </p>
        </div>
    """, unsafe_allow_html=True)


# ============================================
# DATA PATH UTILITIES
# ============================================

def get_data_path(filename):
    """Get path to data file - works locally and on Streamlit Cloud."""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent / "data" / filename,
        Path(__file__).parent / filename,
        Path("data") / filename,
        Path(filename)
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Default to data folder
    return Path("data") / filename


# ============================================
# DATA LOADING FUNCTIONS
# ============================================

@st.cache_data(ttl=3600)
def load_diabetes_data():
    """Load and preprocess diabetes dataset."""
    df = pd.read_csv(get_data_path("diabetes.csv"))

    # Replace zero values with median for columns where 0 is invalid
    zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_not_valid:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    return df


@st.cache_data(ttl=3600)
def load_heart_data():
    """Load and preprocess heart disease dataset."""
    df = pd.read_csv(get_data_path("Heart_Disease_Prediction.csv"))

    # Encode target variable
    if 'Heart Disease' in df.columns:
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    df = df.fillna(df.median(numeric_only=True))
    return df


@st.cache_data(ttl=3600)
def load_parkinsons_data():
    """Load and preprocess Parkinson's dataset."""
    df = pd.read_csv(get_data_path("parkinsons.csv"))

    if 'name' in df.columns:
        df = df.drop('name', axis=1)

    return df


@st.cache_data(ttl=3600)
def load_liver_data():
    """Load and preprocess liver disease dataset."""
    df = pd.read_csv(get_data_path("indian_liver_patient.csv"))

    # Encode Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df = df.fillna(df.median(numeric_only=True))

    # Convert target: 1 = Liver Disease, 2 = No Disease -> 1 = Disease, 0 = No Disease
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

    return df


# ============================================
# MODEL TRAINING FUNCTIONS
# ============================================

@st.cache_resource(ttl=3600)
def train_model(disease_type):
    """Train and cache model for specified disease."""

    if disease_type == "diabetes":
        df = load_diabetes_data()
        X = df.drop('Outcome', axis=1).values
        y = df['Outcome'].values
        feature_names = df.drop('Outcome', axis=1).columns.tolist()

    elif disease_type == "heart":
        df = load_heart_data()
        X = df.drop('Heart Disease', axis=1).values
        y = df['Heart Disease'].values
        feature_names = df.drop('Heart Disease', axis=1).columns.tolist()

    elif disease_type == "parkinsons":
        df = load_parkinsons_data()
        X = df.drop('status', axis=1).values
        y = df['status'].values
        feature_names = df.drop('status', axis=1).columns.tolist()

    elif disease_type == "liver":
        df = load_liver_data()
        X = df.drop('Dataset', axis=1).values
        y = df['Dataset'].values
        feature_names = df.drop('Dataset', axis=1).columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models and select best
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ""
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model = model
            best_name = name

    return best_model, scaler, best_name, results[best_name]['accuracy'], results, feature_names


def make_prediction(model, scaler, features):
    """Make prediction using trained model."""
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features_scaled)[0]
        prob = probability[1] if prediction == 1 else probability[0]
    else:
        prob = 0.5

    return int(prediction), prob


def get_risk_level(probability, prediction):
    """Determine risk level based on prediction."""
    if prediction == 0:
        return "Low"
    else:
        if probability >= 0.8:
            return "High"
        elif probability >= 0.6:
            return "Medium"
        else:
            return "Medium"


# ============================================
# PAGE RENDERING FUNCTIONS
# ============================================

def render_home_page():
    """Render home page."""
    # Logo and Title Section
    col_logo, col_title = st.columns([1, 3])

    with col_logo:
        st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <img src="https://medindia.net/images/common/medindia-logo.png"
                     alt="MedIndia Logo"
                     style="max-width: 150px; height: auto;">
            </div>
        """, unsafe_allow_html=True)

    with col_title:
        st.markdown("""
            <div style="padding: 1rem 0;">
                <h1 style="color: #1e3a5f; margin: 0;">Welcome to MedIndia's HealthPredict AI</h1>
                <p style="color: #64748b; font-size: 1.1rem; margin-top: 0.5rem;">
                    Advanced Machine Learning for Early Disease Detection
                </p>
                <p style="color: #3b82f6; font-size: 0.95rem; font-style: italic;">
                    Empowering Better Health
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Hero section
    st.markdown("""
        <div style="background: linear-gradient(135deg, #c41e3a 0%, #1e3a5f 50%, #3b82f6 100%);
                    border-radius: 20px; padding: 2rem; margin: 1rem 0; color: white;">
            <h2 style="color: white !important;">Your Personal Health Risk Assessment Platform</h2>
            <p style="color: #e0e7ff;">
                Powered by machine learning algorithms, our system analyzes your health
                parameters to provide instant risk assessments for multiple diseases.
                A service by <strong>MedIndia - Empowering Better Health</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Disease cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="health-card">
                <h3>🩸 Diabetes Risk Assessment</h3>
                <p style="color: #64748b;">Evaluate your risk of Type 2 Diabetes based on glucose, BMI, and more.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="health-card">
                <h3>🧠 Parkinson's Screening</h3>
                <p style="color: #64748b;">Voice-based screening using biomedical voice measurements.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="health-card">
                <h3>❤️ Heart Disease Check</h3>
                <p style="color: #64748b;">Cardiac risk assessment analyzing cholesterol, BP, and ECG results.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="health-card">
                <h3>🫁 Liver Health Analysis</h3>
                <p style="color: #64748b;">Liver function assessment based on bilirubin and enzyme levels.</p>
            </div>
        """, unsafe_allow_html=True)

    render_disclaimer()
    render_medindia_footer()


def render_diabetes_page():
    """Render diabetes prediction page."""
    st.markdown("<h1>🩸 Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Evaluate your risk using the PIMA Indians dataset model</p>", unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading model..."):
        model, scaler, model_name, accuracy, results, features = train_model("diabetes")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='health-card'><h3>📋 Enter Your Health Parameters</h3></div>", unsafe_allow_html=True)

        with st.form("diabetes_form"):
            c1, c2 = st.columns(2)
            with c1:
                pregnancies = st.number_input("Pregnancies", 0, 20, 1)
                glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
                skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            with c2:
                insulin = st.number_input("Insulin (μU/mL)", 0, 900, 80)
                bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0)
                dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
                age = st.number_input("Age (years)", 1, 120, 30)

            submitted = st.form_submit_button("🔍 Analyze Risk", use_container_width=True)

    with col2:
        if submitted:
            features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            prediction, probability = make_prediction(model, scaler, features)
            risk_level = get_risk_level(probability, prediction)

            if prediction == 1:
                st.markdown(f"""
                    <div class="result-positive">
                        <span style="font-size: 3rem;">⚠️</span>
                        <h2 style="color: #dc2626;">At Risk for Diabetes</h2>
                        <span class="risk-{risk_level.lower()}">{risk_level} Risk</span>
                        <p style="margin-top: 1rem;">Confidence: {probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-negative">
                        <span style="font-size: 3rem;">✅</span>
                        <h2 style="color: #059669;">Low Risk for Diabetes</h2>
                        <span class="risk-low">Low Risk</span>
                        <p style="margin-top: 1rem;">Confidence: {probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter your health parameters and click 'Analyze Risk'")

    st.markdown("---")
    st.markdown(f"**Model:** {model_name} | **Accuracy:** {accuracy*100:.1f}%")
    render_disclaimer()
    render_medindia_footer()


def render_heart_page():
    """Render heart disease prediction page."""
    st.markdown("<h1>❤️ Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Comprehensive cardiac health evaluation</p>", unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        model, scaler, model_name, accuracy, results, features = train_model("heart")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='health-card'><h3>📋 Enter Cardiac Parameters</h3></div>", unsafe_allow_html=True)

        with st.form("heart_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", 1, 120, 50)
                sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
                chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                    format_func=lambda x: {1: "Typical Angina", 2: "Atypical", 3: "Non-anginal", 4: "Asymptomatic"}[x])
                bp = st.number_input("Resting BP (mm Hg)", 80, 220, 120)
                cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
                fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            with c2:
                ekg = st.selectbox("Resting EKG", [0, 1, 2],
                    format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])
                max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
                exercise_angina = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                st_depression = st.number_input("ST Depression", 0.0, 10.0, 1.0)
                slope = st.selectbox("ST Slope", [1, 2, 3],
                    format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
                vessels = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])

            thallium = st.selectbox("Thallium Test", [3, 6, 7],
                format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

            submitted = st.form_submit_button("🔍 Analyze Cardiac Risk", use_container_width=True)

    with col2:
        if submitted:
            features = [age, sex, chest_pain, bp, cholesterol, fbs, ekg, max_hr,
                       exercise_angina, st_depression, slope, vessels, thallium]
            prediction, probability = make_prediction(model, scaler, features)
            risk_level = get_risk_level(probability, prediction)

            if prediction == 1:
                st.markdown(f"""
                    <div class="result-positive">
                        <span style="font-size: 3rem;">⚠️</span>
                        <h2 style="color: #dc2626;">At Risk for Heart Disease</h2>
                        <span class="risk-{risk_level.lower()}">{risk_level} Risk</span>
                        <p style="margin-top: 1rem;">Confidence: {probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-negative">
                        <span style="font-size: 3rem;">✅</span>
                        <h2 style="color: #059669;">Low Risk for Heart Disease</h2>
                        <span class="risk-low">Low Risk</span>
                        <p style="margin-top: 1rem;">Confidence: {probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter your cardiac parameters and click 'Analyze Cardiac Risk'")

    st.markdown("---")
    st.markdown(f"**Model:** {model_name} | **Accuracy:** {accuracy*100:.1f}%")
    render_disclaimer()
    render_medindia_footer()


def render_parkinsons_page():
    """Render Parkinson's prediction page."""
    st.markdown("<h1>🧠 Parkinson's Disease Screening</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Voice-based screening using biomedical measurements</p>", unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        model, scaler, model_name, accuracy, results, features = train_model("parkinsons")

    st.info("ℹ️ This screening uses voice measurements. Values should be obtained from voice analysis software.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='health-card'><h3>📋 Enter Voice Measurements</h3></div>", unsafe_allow_html=True)

        with st.form("parkinsons_form"):
            st.markdown("**Frequency Parameters:**")
            c1, c2 = st.columns(2)
            with c1:
                fo = st.number_input("MDVP:Fo (Hz)", 80.0, 300.0, 150.0)
                fhi = st.number_input("MDVP:Fhi (Hz)", 100.0, 600.0, 200.0)
                flo = st.number_input("MDVP:Flo (Hz)", 50.0, 250.0, 100.0)
            with c2:
                jitter_percent = st.number_input("Jitter (%)", 0.0, 0.1, 0.005, format="%.5f")
                jitter_abs = st.number_input("Jitter (Abs)", 0.0, 0.001, 0.00005, format="%.6f")
                rap = st.number_input("MDVP:RAP", 0.0, 0.05, 0.003, format="%.5f")

            c3, c4 = st.columns(2)
            with c3:
                ppq = st.number_input("MDVP:PPQ", 0.0, 0.05, 0.003, format="%.5f")
                ddp = st.number_input("Jitter:DDP", 0.0, 0.1, 0.01, format="%.5f")
                shimmer = st.number_input("MDVP:Shimmer", 0.0, 0.2, 0.03, format="%.4f")
                shimmer_db = st.number_input("Shimmer (dB)", 0.0, 2.0, 0.3, format="%.3f")
                apq3 = st.number_input("Shimmer:APQ3", 0.0, 0.1, 0.015, format="%.4f")
            with c4:
                apq5 = st.number_input("Shimmer:APQ5", 0.0, 0.1, 0.02, format="%.4f")
                apq = st.number_input("MDVP:APQ", 0.0, 0.1, 0.02, format="%.4f")
                dda = st.number_input("Shimmer:DDA", 0.0, 0.2, 0.05, format="%.4f")
                nhr = st.number_input("NHR", 0.0, 0.5, 0.02, format="%.4f")
                hnr = st.number_input("HNR", 0.0, 40.0, 22.0)

            c5, c6 = st.columns(2)
            with c5:
                rpde = st.number_input("RPDE", 0.0, 1.0, 0.5, format="%.4f")
                dfa = st.number_input("DFA", 0.5, 1.0, 0.7, format="%.4f")
                spread1 = st.number_input("Spread1", -10.0, 0.0, -5.0, format="%.4f")
            with c6:
                spread2 = st.number_input("Spread2", 0.0, 0.5, 0.25, format="%.4f")
                d2 = st.number_input("D2", 1.0, 4.0, 2.5, format="%.4f")
                ppe = st.number_input("PPE", 0.0, 0.6, 0.2, format="%.4f")

            submitted = st.form_submit_button("🔍 Analyze Voice Patterns", use_container_width=True)

    with col2:
        if submitted:
            features = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                       shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                       rpde, dfa, spread1, spread2, d2, ppe]
            prediction, probability = make_prediction(model, scaler, features)
            risk_level = get_risk_level(probability, prediction)

            if prediction == 1:
                st.markdown(f"""
                    <div class="result-positive">
                        <span style="font-size: 3rem;">⚠️</span>
                        <h2 style="color: #dc2626;">Potential Parkinson's Indicators</h2>
                        <span class="risk-{risk_level.lower()}">{risk_level} Risk</span>
                        <p style="margin-top: 1rem;">Confidence: {probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-negative">
                        <span style="font-size: 3rem;">✅</span>
                        <h2 style="color: #059669;">No Parkinson's Indicators</h2>
                        <span class="risk-low">Low Risk</span>
                        <p style="margin-top: 1rem;">Confidence: {probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter voice measurements and click 'Analyze Voice Patterns'")

    st.markdown("---")
    st.markdown(f"**Model:** {model_name} | **Accuracy:** {accuracy*100:.1f}%")
    render_disclaimer()
    render_medindia_footer()


def render_liver_page():
    """Render liver disease prediction page."""
    st.markdown("<h1>🫁 Liver Health Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b;'>Comprehensive liver function assessment</p>", unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        model, scaler, model_name, accuracy, results, features = train_model("liver")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='health-card'><h3>📋 Enter Liver Function Parameters</h3></div>", unsafe_allow_html=True)

        with st.form("liver_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", 1, 120, 45)
                gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
                total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.0, 80.0, 1.0)
                direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", 0.0, 20.0, 0.3)
                alkaline_phosphotase = st.number_input("Alkaline Phosphatase (IU/L)", 0, 2500, 200)
            with c2:
                alt = st.number_input("ALT / SGPT (IU/L)", 0, 2000, 25)
                ast = st.number_input("AST / SGOT (IU/L)", 0, 5000, 30)
                total_proteins = st.number_input("Total Proteins (g/dL)", 0.0, 15.0, 6.5)
                albumin = st.number_input("Albumin (g/dL)", 0.0, 10.0, 3.5)
                ag_ratio = st.number_input("A/G Ratio", 0.0, 3.0, 1.0)

            submitted = st.form_submit_button("🔍 Analyze Liver Health", use_container_width=True)

    with col2:
        if submitted:
            features = [age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                       alt, ast, total_proteins, albumin, ag_ratio]
            prediction, probability = make_prediction(model, scaler, features)
            risk_level = get_risk_level(probability, prediction)

            if prediction == 1:
                st.markdown(f"""
                    <div class="result-positive">
                        <span style="font-size: 3rem;">⚠️</span>
                        <h2 style="color: #dc2626;">At Risk for Liver Disease</h2>
                        <span class="risk-{risk_level.lower()}">{risk_level} Risk</span>
                        <p style="margin-top: 1rem;">Confidence: {probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-negative">
                        <span style="font-size: 3rem;">✅</span>
                        <h2 style="color: #059669;">Low Risk for Liver Disease</h2>
                        <span class="risk-low">Low Risk</span>
                        <p style="margin-top: 1rem;">Confidence: {probability*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter liver function parameters and click 'Analyze Liver Health'")

    st.markdown("---")
    st.markdown(f"**Model:** {model_name} | **Accuracy:** {accuracy*100:.1f}%")
    render_disclaimer()
    render_medindia_footer()


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="MedIndia's HealthPredict AI - Multi-Disease Prediction",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_custom_styles()

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'

    # Sidebar
    with st.sidebar:
        # MedIndia Logo
        st.markdown("""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; margin-bottom: 1rem;">
                <img src="https://medindia.net/images/common/medindia-logo.png"
                     alt="MedIndia Logo"
                     style="max-width: 180px; height: auto;">
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align: center; padding: 0.5rem;">
                <h2 style="color: #1e3a5f; font-size: 1.3rem; margin: 0;">
                    HealthPredict AI
                </h2>
                <p style="color: #64748b; font-size: 0.85rem; margin: 0.5rem 0 0 0;">
                    Multi-Disease Prediction System
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        menu_items = {
            'Home': '🏠 Home',
            'Diabetes': '🩸 Diabetes',
            'Heart Disease': '❤️ Heart Disease',
            'Parkinsons': '🧠 Parkinsons',
            'Liver Disease': '🫁 Liver Disease'
        }

        for page, label in menu_items.items():
            if st.button(label, key=f"nav_{page}", use_container_width=True,
                        type="primary" if st.session_state.current_page == page else "secondary"):
                st.session_state.current_page = page
                st.rerun()

        st.markdown("---")
        st.markdown("""
            <div style="padding: 1rem; background: #f0f9ff; border-radius: 10px;">
                <p style="color: #0369a1; font-size: 0.8rem;">
                    ℹ️ This tool is for informational purposes only.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # MedIndia Footer
        st.markdown("""
            <div style="text-align: center; padding: 1rem; margin-top: 1rem;">
                <p style="color: #64748b; font-size: 0.75rem;">
                    © 2024 MedIndia<br>
                    <span style="color: #3b82f6;">Empowering Better Health</span>
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Route to pages
    if st.session_state.current_page == 'Home':
        render_home_page()
    elif st.session_state.current_page == 'Diabetes':
        render_diabetes_page()
    elif st.session_state.current_page == 'Heart Disease':
        render_heart_page()
    elif st.session_state.current_page == 'Parkinsons':
        render_parkinsons_page()
    elif st.session_state.current_page == 'Liver Disease':
        render_liver_page()


if __name__ == "__main__":
    main()
