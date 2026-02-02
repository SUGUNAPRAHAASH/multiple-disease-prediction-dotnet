# HealthPredict AI - Setup Guide

## Multi-Disease Prediction System
### ASP.NET Core MVC Frontend + Flask REST API Backend

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER BROWSER                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              ASP.NET Core MVC Frontend                      │
│              (Port 5001 / HTTPS: 7001)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Healthcare-grade UI                               │   │
│  │  • Clickable Disease Cards                          │   │
│  │  • Form Validation                                  │   │
│  │  • Result Display                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                    REST API (JSON)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Flask REST API Backend                       │
│                     (Port 5000)                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • ML Model Loading                                 │   │
│  │  • Prediction Endpoints                             │   │
│  │  • Feature Scaling                                  │   │
│  │  • Risk Assessment                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ │
│  │ Diabetes  │  │  Heart    │  │Parkinson's│  │  Liver   │ │
│  │  Model    │  │  Model    │  │   Model   │  │  Model   │ │
│  └───────────┘  └───────────┘  └───────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### For Flask API Backend
- Python 3.9 or higher
- pip (Python package manager)

### For ASP.NET Core MVC Frontend
- .NET 8.0 SDK or higher
- Visual Studio 2022 or VS Code with C# extension

---

## Installation Steps

### Step 1: Set Up Flask API Backend

1. **Navigate to Flask API directory:**
   ```bash
   cd "E:/disease -net/flask_api"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Flask API server:**
   ```bash
   python app.py
   ```

   The API will start on `http://localhost:5000`

   You should see:
   ```
   Loading models...
     - diabetes model loaded successfully
     - heart model loaded successfully
     - parkinsons model loaded successfully
     - liver model loaded successfully

   Starting Flask API server...
    * Running on http://0.0.0.0:5000
   ```

### Step 2: Set Up ASP.NET Core MVC Frontend

1. **Navigate to MVC project directory:**
   ```bash
   cd "E:/disease -net/HealthPredictMVC"
   ```

2. **Restore NuGet packages:**
   ```bash
   dotnet restore
   ```

3. **Build the project:**
   ```bash
   dotnet build
   ```

4. **Run the application:**
   ```bash
   dotnet run
   ```

   The frontend will start on:
   - HTTP: `http://localhost:5001`
   - HTTPS: `https://localhost:7001`

5. **Open your browser and navigate to:**
   ```
   https://localhost:7001
   ```

---

## API Endpoints Reference

### Health Check
```
GET /api/health
```
Response:
```json
{
  "status": "healthy",
  "service": "HealthPredict AI API",
  "version": "1.0.0"
}
```

### Diabetes Prediction
```
POST /api/diabetes/predict
Content-Type: application/json

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
```

### Heart Disease Prediction
```
POST /api/heart/predict
Content-Type: application/json

{
  "Age": 55,
  "Sex": 1,
  "Chest pain type": 2,
  "BP": 140,
  "Cholesterol": 250,
  "FBS over 120": 0,
  "EKG results": 0,
  "Max HR": 150,
  "Exercise angina": 0,
  "ST depression": 1.5,
  "Slope of ST": 2,
  "Number of vessels fluro": 0,
  "Thallium": 3
}
```

### Parkinson's Prediction
```
POST /api/parkinsons/predict
Content-Type: application/json

{
  "MDVP:Fo(Hz)": 120,
  "MDVP:Fhi(Hz)": 150,
  "MDVP:Flo(Hz)": 100,
  ... (22 voice parameters)
}
```

### Liver Disease Prediction
```
POST /api/liver/predict
Content-Type: application/json

{
  "Age": 45,
  "Gender": 1,
  "Total_Bilirubin": 1.0,
  "Direct_Bilirubin": 0.3,
  "Alkaline_Phosphotase": 180,
  "Alamine_Aminotransferase": 25,
  "Aspartate_Aminotransferase": 30,
  "Total_Protiens": 7.0,
  "Albumin": 4.0,
  "Albumin_and_Globulin_Ratio": 1.1
}
```

---

## Project Structure

```
E:/disease -net/
│
├── flask_api/                      # Flask REST API Backend
│   ├── app.py                      # Main Flask application
│   └── requirements.txt            # Python dependencies
│
├── HealthPredictMVC/               # ASP.NET Core MVC Frontend
│   ├── Controllers/
│   │   ├── HomeController.cs       # Homepage controller
│   │   ├── DiabetesController.cs   # Diabetes prediction
│   │   ├── HeartDiseaseController.cs
│   │   ├── ParkinsonsController.cs
│   │   └── LiverController.cs
│   │
│   ├── Models/
│   │   └── PredictionModels.cs     # All input/output models
│   │
│   ├── Services/
│   │   └── PredictionService.cs    # API integration service
│   │
│   ├── Views/
│   │   ├── Home/
│   │   │   └── Index.cshtml        # Homepage with clickable cards
│   │   ├── Diabetes/
│   │   │   └── Index.cshtml        # Diabetes assessment page
│   │   ├── HeartDisease/
│   │   │   └── Index.cshtml
│   │   ├── Parkinsons/
│   │   │   └── Index.cshtml
│   │   ├── Liver/
│   │   │   └── Index.cshtml
│   │   └── Shared/
│   │       ├── _Layout.cshtml      # Main layout template
│   │       └── Error.cshtml
│   │
│   ├── wwwroot/
│   │   └── css/
│   │       └── site.css            # Healthcare-grade CSS
│   │
│   ├── Program.cs                  # Application entry point
│   ├── appsettings.json            # Configuration
│   └── HealthPredictMVC.csproj     # Project file
│
├── models/                         # Pre-trained ML models
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── parkinsons_model.pkl
│   └── liver_model.pkl
│
├── data/                           # Training datasets
│   ├── diabetes.csv
│   ├── Heart_Disease_Prediction.csv
│   ├── parkinsons.csv
│   └── indian_liver_patient.csv
│
└── app.py                          # Original Streamlit app (backup)
```

---

## Configuration

### Flask API URL Configuration

In `HealthPredictMVC/appsettings.json`:
```json
{
  "FlaskApiUrl": "http://localhost:5000"
}
```

For production, update this to your deployed API URL.

---

## Running in Production

### Flask API with Gunicorn (Linux/macOS)
```bash
cd flask_api
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Flask API on Windows (Waitress)
```bash
pip install waitress
waitress-serve --port=5000 app:app
```

### ASP.NET Core MVC
```bash
cd HealthPredictMVC
dotnet publish -c Release
cd bin/Release/net8.0/publish
dotnet HealthPredictMVC.dll
```

---

## Troubleshooting

### Issue: API Connection Failed
- Ensure Flask API is running on port 5000
- Check if the `FlaskApiUrl` in `appsettings.json` is correct
- Verify no firewall is blocking the connection

### Issue: Model Not Found
- The Flask API will automatically train models if pre-trained models aren't found
- Ensure the `data/` folder contains the CSV files
- Check that the `models/` folder is writable

### Issue: Port Already in Use
- Change Flask port: `app.run(port=5001)`
- Change ASP.NET port in `launchSettings.json`

---

## Security Considerations

1. **HTTPS**: Always use HTTPS in production
2. **CORS**: Configure CORS appropriately for your domains
3. **API Keys**: Consider adding API key authentication for production
4. **Rate Limiting**: Implement rate limiting to prevent abuse
5. **Input Validation**: Both frontend and backend validate inputs

---

## Features Implemented

✅ **Clickable Homepage Cards** - Each disease card navigates to its assessment page
✅ **Disease Overview Section** - Medical explanation at the top of each page
✅ **Feature Explanations** - Each input parameter explained before the form
✅ **Healthcare-grade UI** - Professional, responsive design
✅ **REST API Integration** - Secure JSON communication
✅ **Form Validation** - Client and server-side validation
✅ **Risk Level Assessment** - Low/Medium/High risk categorization
✅ **Medical Disclaimers** - Prominent on all pages

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | ASP.NET Core 8.0 MVC |
| Backend API | Flask (Python) |
| ML Library | scikit-learn |
| CSS Framework | Bootstrap 5 + Custom CSS |
| Icons | Font Awesome 6 |

---

## License

This project is for educational and informational purposes only.

**Medical Disclaimer**: This application provides informational insights only and is not a medical diagnosis. Always consult qualified healthcare professionals.

---

© 2024 MedIndia - Empowering Better Health
