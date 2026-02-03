"""
Flask REST API for Multi-Disease Prediction System
HealthPredict AI by MedIndia

This API provides endpoints for:
- Diabetes Risk Assessment
- Heart Disease Check
- Parkinson's Screening
- Liver Health Analysis
- Knee Osteoarthritis Assessment (CNN-based image analysis)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json
import base64
from io import BytesIO
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

app = Flask(__name__)
CORS(app)  # Enable CORS for ASP.NET Core frontend


@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'app': 'HealthPredict AI - Multi-Disease Prediction API',
        'endpoints': [
            '/api/diabetes/predict',
            '/api/heart/predict',
            '/api/parkinsons/predict',
            '/api/liver/predict',
            '/api/knee/predict'
        ]
    })


@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy'})


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Cache for loaded models
_model_cache = {}


def get_model_path(disease_type, file_type):
    """Get path for model files."""
    return os.path.join(MODELS_DIR, f'{disease_type}_{file_type}.pkl')


def load_model(disease_type):
    """Load pre-trained model and scaler from disk."""
    if disease_type in _model_cache:
        return _model_cache[disease_type]

    try:
        model = joblib.load(get_model_path(disease_type, 'model'))
        scaler = joblib.load(get_model_path(disease_type, 'scaler'))
        features = joblib.load(get_model_path(disease_type, 'features'))
        info = joblib.load(get_model_path(disease_type, 'info'))

        _model_cache[disease_type] = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'info': info
        }
        return _model_cache[disease_type]
    except Exception as e:
        print(f"Error loading model for {disease_type}: {e}")
        # Train model if not found
        return train_model_fallback(disease_type)


def load_diabetes_data():
    """Load and preprocess diabetes dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'diabetes.csv'))

    # Replace zeros with median for certain columns
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        if col in df.columns:
            median_val = df[df[col] != 0][col].median()
            df[col] = df[col].replace(0, median_val)

    return df


def load_heart_data():
    """Load and preprocess heart disease dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'Heart_Disease_Prediction.csv'))

    # Map target column if needed
    if 'Heart Disease' in df.columns:
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
        if df['Heart Disease'].isna().any():
            df['Heart Disease'] = pd.to_numeric(df['Heart Disease'], errors='coerce')

    return df


def load_parkinsons_data():
    """Load and preprocess Parkinson's dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'parkinsons.csv'))

    # Remove name column if present
    if 'name' in df.columns:
        df = df.drop('name', axis=1)

    return df


def load_liver_data():
    """Load and preprocess liver disease dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'indian_liver_patient.csv'))

    # Encode gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # Map target column
    if 'Dataset' in df.columns:
        df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

    return df


def train_model_fallback(disease_type):
    """Train model if pre-trained model not available."""
    print(f"Training {disease_type} model...")

    # Load appropriate data
    if disease_type == 'diabetes':
        df = load_diabetes_data()
        target_col = 'Outcome'
    elif disease_type == 'heart':
        df = load_heart_data()
        target_col = 'Heart Disease'
    elif disease_type == 'parkinsons':
        df = load_parkinsons_data()
        target_col = 'status'
    elif disease_type == 'liver':
        df = load_liver_data()
        target_col = 'Dataset'
    else:
        raise ValueError(f"Unknown disease type: {disease_type}")

    # Prepare features
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    features = list(X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models and select best
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Add SVM for Parkinson's
    if disease_type == 'parkinsons':
        models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)

    best_model = None
    best_score = 0
    best_name = ''
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        mean_score = cv_scores.mean()
        results[name] = {
            'cv_mean': mean_score,
            'cv_std': cv_scores.std(),
            'test_score': model.score(X_test_scaled, y_test)
        }

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, get_model_path(disease_type, 'model'))
    joblib.dump(scaler, get_model_path(disease_type, 'scaler'))
    joblib.dump(features, get_model_path(disease_type, 'features'))
    joblib.dump({'model_name': best_name, 'results': results}, get_model_path(disease_type, 'info'))

    _model_cache[disease_type] = {
        'model': best_model,
        'scaler': scaler,
        'features': features,
        'info': {'model_name': best_name, 'results': results}
    }

    return _model_cache[disease_type]


def make_prediction(disease_type, features_dict):
    """Make prediction using loaded model."""
    model_data = load_model(disease_type)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['features']

    # Create feature array in correct order
    feature_array = np.array([[features_dict.get(f, 0) for f in feature_names]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Make prediction
    prediction = model.predict(scaled_features)[0]

    # Get probability
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_features)[0]
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
    else:
        probability = float(prediction)

    return int(prediction), float(probability)


def get_risk_level(prediction, probability):
    """Determine risk level based on prediction and probability."""
    if prediction == 0:
        return "Low"
    else:
        if probability >= 0.8:
            return "High"
        elif probability >= 0.6:
            return "Medium"
        else:
            return "Medium"


# ============== API ENDPOINTS ==============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'HealthPredict AI API',
        'version': '1.0.0'
    })


@app.route('/api/diabetes/predict', methods=['POST'])
def predict_diabetes():
    """
    Diabetes Risk Assessment API

    Expected JSON body:
    {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        prediction, probability = make_prediction('diabetes', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': 'High risk of diabetes detected. Please consult a healthcare professional.'
                      if prediction == 1
                      else 'Low risk of diabetes. Maintain a healthy lifestyle.',
            'disease': 'Diabetes'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/heart/predict', methods=['POST'])
def predict_heart():
    """
    Heart Disease Risk Assessment API

    Expected JSON body:
    {
        "Age": 55,
        "Sex": 1,
        "Chest pain type": 2,
        "BP": 140,
        "Cholesterol": 250,
        "FBS over 120": 1,
        "EKG results": 0,
        "Max HR": 150,
        "Exercise angina": 0,
        "ST depression": 1.5,
        "Slope of ST": 2,
        "Number of vessels fluro": 1,
        "Thallium": 3
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
                          'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                          'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        prediction, probability = make_prediction('heart', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': 'Indicators suggest potential heart disease risk. Please seek medical evaluation.'
                      if prediction == 1
                      else 'No immediate heart disease risk detected. Continue healthy habits.',
            'disease': 'Heart Disease'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/parkinsons/predict', methods=['POST'])
def predict_parkinsons():
    """
    Parkinson's Disease Screening API

    Expected JSON body with 22 voice biomarker parameters.
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        prediction, probability = make_prediction('parkinsons', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': "Voice patterns suggest potential Parkinson's indicators. Consult a neurologist."
                      if prediction == 1
                      else "Voice patterns appear normal. No Parkinson's indicators detected.",
            'disease': "Parkinson's Disease"
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/liver/predict', methods=['POST'])
def predict_liver():
    """
    Liver Health Analysis API

    Expected JSON body:
    {
        "Age": 45,
        "Gender": 1,
        "Total_Bilirubin": 1.5,
        "Direct_Bilirubin": 0.5,
        "Alkaline_Phosphotase": 200,
        "Alamine_Aminotransferase": 30,
        "Aspartate_Aminotransferase": 35,
        "Total_Protiens": 7.0,
        "Albumin": 4.0,
        "Albumin_and_Globulin_Ratio": 1.2
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                          'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                          'Aspartate_Aminotransferase', 'Total_Protiens',
                          'Albumin', 'Albumin_and_Globulin_Ratio']

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        prediction, probability = make_prediction('liver', data)
        risk_level = get_risk_level(prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'message': 'Liver function indicators suggest potential concern. Please consult a hepatologist.'
                      if prediction == 1
                      else 'Liver function indicators appear normal. Maintain a healthy lifestyle.',
            'disease': 'Liver Disease'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model/info/<disease_type>', methods=['GET'])
def get_model_info(disease_type):
    """Get information about a specific model."""
    valid_types = ['diabetes', 'heart', 'parkinsons', 'liver']

    if disease_type not in valid_types:
        return jsonify({
            'success': False,
            'error': f'Invalid disease type. Must be one of: {valid_types}'
        }), 400

    try:
        model_data = load_model(disease_type)
        info = model_data['info']

        return jsonify({
            'success': True,
            'disease_type': disease_type,
            'model_name': info.get('model_name', 'Unknown'),
            'features': model_data['features'],
            'feature_count': len(model_data['features'])
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============== KNEE OA CNN MODEL ==============

# Knee OA model cache
_knee_model = None
_knee_model_info = None

# Knee OA class information
KNEE_CLASS_NAMES = {
    0: "Normal (Grade 0)",
    1: "Doubtful (Grade 1)",
    2: "Mild (Grade 2)",
    3: "Moderate (Grade 3)",
    4: "Severe (Grade 4)"
}

KNEE_SEVERITY_DESCRIPTIONS = {
    0: "No signs of osteoarthritis. The knee joint appears healthy with normal cartilage space.",
    1: "Doubtful narrowing of joint space with possible osteophytic lipping. Very early signs that may or may not indicate OA.",
    2: "Definite osteophytes and possible narrowing of joint space. Mild osteoarthritis is present.",
    3: "Moderate multiple osteophytes, definite narrowing of joint space, some sclerosis. Moderate OA requiring attention.",
    4: "Large osteophytes, marked narrowing of joint space, severe sclerosis and definite deformity. Severe OA - medical intervention needed."
}

KNEE_RISK_LEVELS = {
    0: "Low",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "High"
}

KNEE_RECOMMENDATIONS = {
    0: "Continue regular physical activity and maintain a healthy weight to protect your knee joints.",
    1: "Consider lifestyle modifications including weight management and low-impact exercises. Regular monitoring recommended.",
    2: "Consult an orthopedic specialist. Physical therapy, anti-inflammatory medications, and lifestyle changes may help.",
    3: "Medical intervention recommended. Treatment options include physical therapy, medications, injections, or possibly surgery.",
    4: "Urgent orthopedic consultation needed. Severe OA may require surgical intervention such as knee replacement."
}


def load_knee_model():
    """Load the trained Knee OA CNN model."""
    global _knee_model, _knee_model_info

    if _knee_model is not None:
        return _knee_model, _knee_model_info

    model_path = os.path.join(MODELS_DIR, 'knee_oa_cnn.keras')
    info_path = os.path.join(MODELS_DIR, 'knee_oa_cnn_info.json')

    try:
        # Import TensorFlow only when needed
        import tensorflow as tf

        if os.path.exists(model_path):
            _knee_model = tf.keras.models.load_model(model_path)
            print("Knee OA CNN model loaded successfully")

            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    _knee_model_info = json.load(f)
            else:
                _knee_model_info = {
                    'model_name': 'Knee OA CNN',
                    'input_shape': (224, 224, 3),
                    'num_classes': 5
                }

            return _knee_model, _knee_model_info
        else:
            raise FileNotFoundError(f"Knee OA model not found at {model_path}. Please train the model first.")

    except Exception as e:
        print(f"Error loading Knee OA model: {e}")
        raise


def preprocess_knee_image(image_data, is_base64=True):
    """Preprocess knee X-ray image for prediction."""
    try:
        if is_base64:
            # Decode base64 image
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        else:
            image = Image.open(image_data)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


@app.route('/api/knee/predict', methods=['POST'])
def predict_knee():
    """
    Knee Osteoarthritis Severity Assessment API

    Accepts:
    - JSON with base64 encoded image: {"image": "base64_string"}
    - Form data with file upload: file field named "image"

    Returns:
    - Severity grade (0-4)
    - Classification name
    - Description
    - Risk level
    - Recommendations
    - Confidence scores for all classes
    """
    try:
        # Load model
        model, model_info = load_knee_model()

        # Get image data
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing image data. Send base64 encoded image in "image" field.'
                }), 400
            image_data = data['image']
            img_array = preprocess_knee_image(image_data, is_base64=True)

        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            img_array = preprocess_knee_image(file, is_base64=False)

        else:
            return jsonify({
                'success': False,
                'error': 'No image provided. Send base64 JSON or file upload.'
            }), 400

        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]

        # Get predicted class and confidence
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class]) * 100

        # Get all class probabilities
        class_probabilities = {
            KNEE_CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(5)
        }

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'grade': f"Grade {predicted_class}",
            'classification': KNEE_CLASS_NAMES[predicted_class],
            'description': KNEE_SEVERITY_DESCRIPTIONS[predicted_class],
            'risk_level': KNEE_RISK_LEVELS[predicted_class],
            'recommendation': KNEE_RECOMMENDATIONS[predicted_class],
            'confidence': round(confidence, 2),
            'all_probabilities': class_probabilities,
            'disease': 'Knee Osteoarthritis'
        })

    except FileNotFoundError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'The CNN model has not been trained yet. Please run train_knee_cnn.py first.'
        }), 503

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/knee/info', methods=['GET'])
def get_knee_model_info():
    """Get information about the Knee OA model."""
    return jsonify({
        'success': True,
        'model_name': 'Knee Osteoarthritis CNN',
        'model_type': 'Convolutional Neural Network (MobileNetV2 Transfer Learning)',
        'input_type': 'X-ray Image',
        'input_size': '224x224 pixels',
        'num_classes': 5,
        'classes': KNEE_CLASS_NAMES,
        'grading_system': 'Kellgren-Lawrence (KL) Scale',
        'severity_descriptions': KNEE_SEVERITY_DESCRIPTIONS,
        'risk_levels': KNEE_RISK_LEVELS
    })


if __name__ == '__main__':
    # Pre-load all models on startup
    print("Loading models...")
    for disease in ['diabetes', 'heart', 'parkinsons', 'liver']:
        try:
            load_model(disease)
            print(f"  - {disease} model loaded successfully")
        except Exception as e:
            print(f"  - {disease} model: {e}")

    # Try to load Knee OA model
    try:
        load_knee_model()
        print("  - knee OA CNN model loaded successfully")
    except Exception as e:
        print(f"  - knee OA CNN model: {e}")

    print("\nStarting Flask API server...")
    app.run(host='0.0.0.0', port=5050, debug=True)
