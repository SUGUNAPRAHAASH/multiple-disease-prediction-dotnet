using System.ComponentModel.DataAnnotations;

namespace HealthPredictMVC.Models
{
    /// <summary>
    /// Base class for prediction results
    /// </summary>
    public class PredictionResult
    {
        public bool Success { get; set; }
        public int Prediction { get; set; }
        public double Probability { get; set; }
        public string RiskLevel { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string Disease { get; set; } = string.Empty;
        public string? Error { get; set; }
    }

    /// <summary>
    /// Diabetes Risk Assessment Input Model
    /// </summary>
    public class DiabetesInput
    {
        [Display(Name = "Number of Pregnancies")]
        [Range(0, 20, ErrorMessage = "Pregnancies must be between 0 and 20")]
        public int Pregnancies { get; set; } = 1;

        [Display(Name = "Glucose Level (mg/dL)")]
        [Range(0, 300, ErrorMessage = "Glucose must be between 0 and 300 mg/dL")]
        public double Glucose { get; set; } = 120;

        [Display(Name = "Blood Pressure (mm Hg)")]
        [Range(0, 200, ErrorMessage = "Blood Pressure must be between 0 and 200 mm Hg")]
        public double BloodPressure { get; set; } = 70;

        [Display(Name = "Skin Thickness (mm)")]
        [Range(0, 100, ErrorMessage = "Skin Thickness must be between 0 and 100 mm")]
        public double SkinThickness { get; set; } = 20;

        [Display(Name = "Insulin Level (μU/mL)")]
        [Range(0, 900, ErrorMessage = "Insulin must be between 0 and 900 μU/mL")]
        public double Insulin { get; set; } = 80;

        [Display(Name = "Body Mass Index (kg/m²)")]
        [Range(0, 70, ErrorMessage = "BMI must be between 0 and 70 kg/m²")]
        public double BMI { get; set; } = 25.0;

        [Display(Name = "Diabetes Pedigree Function")]
        [Range(0, 3, ErrorMessage = "Diabetes Pedigree Function must be between 0 and 3")]
        public double DiabetesPedigreeFunction { get; set; } = 0.5;

        [Display(Name = "Age (years)")]
        [Range(1, 120, ErrorMessage = "Age must be between 1 and 120 years")]
        public int Age { get; set; } = 30;
    }

    /// <summary>
    /// Heart Disease Risk Assessment Input Model
    /// </summary>
    public class HeartDiseaseInput
    {
        [Display(Name = "Age (years)")]
        [Range(1, 120, ErrorMessage = "Age must be between 1 and 120 years")]
        public int Age { get; set; } = 55;

        [Display(Name = "Sex")]
        [Range(0, 1, ErrorMessage = "Sex must be 0 (Female) or 1 (Male)")]
        public int Sex { get; set; } = 1;

        [Display(Name = "Chest Pain Type")]
        [Range(1, 4, ErrorMessage = "Chest Pain Type must be between 1 and 4")]
        public int ChestPainType { get; set; } = 2;

        [Display(Name = "Resting Blood Pressure (mm Hg)")]
        [Range(80, 220, ErrorMessage = "BP must be between 80 and 220 mm Hg")]
        public int BP { get; set; } = 140;

        [Display(Name = "Cholesterol (mg/dL)")]
        [Range(100, 600, ErrorMessage = "Cholesterol must be between 100 and 600 mg/dL")]
        public int Cholesterol { get; set; } = 250;

        [Display(Name = "Fasting Blood Sugar > 120 mg/dL")]
        [Range(0, 1, ErrorMessage = "FBS over 120 must be 0 (No) or 1 (Yes)")]
        public int FBSOver120 { get; set; } = 0;

        [Display(Name = "Resting ECG Results")]
        [Range(0, 2, ErrorMessage = "EKG Results must be 0, 1, or 2")]
        public int EKGResults { get; set; } = 0;

        [Display(Name = "Maximum Heart Rate (bpm)")]
        [Range(60, 220, ErrorMessage = "Max HR must be between 60 and 220 bpm")]
        public int MaxHR { get; set; } = 150;

        [Display(Name = "Exercise Induced Angina")]
        [Range(0, 1, ErrorMessage = "Exercise Angina must be 0 (No) or 1 (Yes)")]
        public int ExerciseAngina { get; set; } = 0;

        [Display(Name = "ST Depression")]
        [Range(0, 10, ErrorMessage = "ST Depression must be between 0 and 10")]
        public double STDepression { get; set; } = 1.5;

        [Display(Name = "Slope of ST Segment")]
        [Range(1, 3, ErrorMessage = "Slope of ST must be 1, 2, or 3")]
        public int SlopeOfST { get; set; } = 2;

        [Display(Name = "Number of Major Vessels (0-3)")]
        [Range(0, 3, ErrorMessage = "Number of vessels must be between 0 and 3")]
        public int NumberOfVessels { get; set; } = 0;

        [Display(Name = "Thallium Stress Test Result")]
        [Range(3, 7, ErrorMessage = "Thallium must be 3, 6, or 7")]
        public int Thallium { get; set; } = 3;
    }

    /// <summary>
    /// Parkinson's Disease Screening Input Model
    /// </summary>
    public class ParkinsonsInput
    {
        // Frequency Parameters
        [Display(Name = "MDVP:Fo (Hz) - Average Vocal Frequency")]
        [Range(50, 300, ErrorMessage = "MDVP:Fo must be between 50 and 300 Hz")]
        public double MDVP_Fo { get; set; } = 120;

        [Display(Name = "MDVP:Fhi (Hz) - Maximum Vocal Frequency")]
        [Range(50, 600, ErrorMessage = "MDVP:Fhi must be between 50 and 600 Hz")]
        public double MDVP_Fhi { get; set; } = 150;

        [Display(Name = "MDVP:Flo (Hz) - Minimum Vocal Frequency")]
        [Range(50, 300, ErrorMessage = "MDVP:Flo must be between 50 and 300 Hz")]
        public double MDVP_Flo { get; set; } = 100;

        // Jitter Parameters
        [Display(Name = "MDVP:Jitter (%) - Frequency Variation")]
        [Range(0, 2, ErrorMessage = "Jitter must be between 0 and 2%")]
        public double MDVP_Jitter_Percent { get; set; } = 0.005;

        [Display(Name = "MDVP:Jitter (Abs) - Absolute Jitter")]
        [Range(0, 0.001, ErrorMessage = "Absolute Jitter must be between 0 and 0.001")]
        public double MDVP_Jitter_Abs { get; set; } = 0.00003;

        [Display(Name = "MDVP:RAP - Relative Average Perturbation")]
        [Range(0, 0.1, ErrorMessage = "RAP must be between 0 and 0.1")]
        public double MDVP_RAP { get; set; } = 0.003;

        [Display(Name = "MDVP:PPQ - Period Perturbation Quotient")]
        [Range(0, 0.1, ErrorMessage = "PPQ must be between 0 and 0.1")]
        public double MDVP_PPQ { get; set; } = 0.003;

        [Display(Name = "Jitter:DDP - Differential Jitter")]
        [Range(0, 0.1, ErrorMessage = "DDP must be between 0 and 0.1")]
        public double Jitter_DDP { get; set; } = 0.008;

        // Shimmer Parameters
        [Display(Name = "MDVP:Shimmer - Amplitude Variation")]
        [Range(0, 0.2, ErrorMessage = "Shimmer must be between 0 and 0.2")]
        public double MDVP_Shimmer { get; set; } = 0.03;

        [Display(Name = "MDVP:Shimmer (dB)")]
        [Range(0, 2, ErrorMessage = "Shimmer dB must be between 0 and 2")]
        public double MDVP_Shimmer_dB { get; set; } = 0.3;

        [Display(Name = "Shimmer:APQ3")]
        [Range(0, 0.1, ErrorMessage = "APQ3 must be between 0 and 0.1")]
        public double Shimmer_APQ3 { get; set; } = 0.015;

        [Display(Name = "Shimmer:APQ5")]
        [Range(0, 0.2, ErrorMessage = "APQ5 must be between 0 and 0.2")]
        public double Shimmer_APQ5 { get; set; } = 0.02;

        [Display(Name = "MDVP:APQ")]
        [Range(0, 0.2, ErrorMessage = "APQ must be between 0 and 0.2")]
        public double MDVP_APQ { get; set; } = 0.025;

        [Display(Name = "Shimmer:DDA")]
        [Range(0, 0.2, ErrorMessage = "DDA must be between 0 and 0.2")]
        public double Shimmer_DDA { get; set; } = 0.045;

        // Noise Parameters
        [Display(Name = "NHR - Noise-to-Harmonics Ratio")]
        [Range(0, 1, ErrorMessage = "NHR must be between 0 and 1")]
        public double NHR { get; set; } = 0.025;

        [Display(Name = "HNR - Harmonics-to-Noise Ratio")]
        [Range(0, 40, ErrorMessage = "HNR must be between 0 and 40")]
        public double HNR { get; set; } = 22;

        // Nonlinear Parameters
        [Display(Name = "RPDE - Recurrence Period Density Entropy")]
        [Range(0, 1, ErrorMessage = "RPDE must be between 0 and 1")]
        public double RPDE { get; set; } = 0.5;

        [Display(Name = "DFA - Detrended Fluctuation Analysis")]
        [Range(0, 1, ErrorMessage = "DFA must be between 0 and 1")]
        public double DFA { get; set; } = 0.7;

        [Display(Name = "Spread1")]
        [Range(-10, 0, ErrorMessage = "Spread1 must be between -10 and 0")]
        public double Spread1 { get; set; } = -5;

        [Display(Name = "Spread2")]
        [Range(0, 1, ErrorMessage = "Spread2 must be between 0 and 1")]
        public double Spread2 { get; set; } = 0.2;

        [Display(Name = "D2 - Correlation Dimension")]
        [Range(0, 5, ErrorMessage = "D2 must be between 0 and 5")]
        public double D2 { get; set; } = 2.5;

        [Display(Name = "PPE - Pitch Period Entropy")]
        [Range(0, 1, ErrorMessage = "PPE must be between 0 and 1")]
        public double PPE { get; set; } = 0.2;
    }

    /// <summary>
    /// Liver Health Analysis Input Model
    /// </summary>
    public class LiverInput
    {
        [Display(Name = "Age (years)")]
        [Range(1, 120, ErrorMessage = "Age must be between 1 and 120 years")]
        public int Age { get; set; } = 45;

        [Display(Name = "Gender")]
        [Range(0, 1, ErrorMessage = "Gender must be 0 (Female) or 1 (Male)")]
        public int Gender { get; set; } = 1;

        [Display(Name = "Total Bilirubin (mg/dL)")]
        [Range(0, 80, ErrorMessage = "Total Bilirubin must be between 0 and 80 mg/dL")]
        public double TotalBilirubin { get; set; } = 1.0;

        [Display(Name = "Direct Bilirubin (mg/dL)")]
        [Range(0, 20, ErrorMessage = "Direct Bilirubin must be between 0 and 20 mg/dL")]
        public double DirectBilirubin { get; set; } = 0.3;

        [Display(Name = "Alkaline Phosphatase (IU/L)")]
        [Range(0, 2500, ErrorMessage = "Alkaline Phosphatase must be between 0 and 2500 IU/L")]
        public int AlkalinePhosphatase { get; set; } = 180;

        [Display(Name = "ALT / SGPT (IU/L)")]
        [Range(0, 2000, ErrorMessage = "ALT must be between 0 and 2000 IU/L")]
        public int ALT { get; set; } = 25;

        [Display(Name = "AST / SGOT (IU/L)")]
        [Range(0, 5000, ErrorMessage = "AST must be between 0 and 5000 IU/L")]
        public int AST { get; set; } = 30;

        [Display(Name = "Total Proteins (g/dL)")]
        [Range(0, 15, ErrorMessage = "Total Proteins must be between 0 and 15 g/dL")]
        public double TotalProteins { get; set; } = 7.0;

        [Display(Name = "Albumin (g/dL)")]
        [Range(0, 10, ErrorMessage = "Albumin must be between 0 and 10 g/dL")]
        public double Albumin { get; set; } = 4.0;

        [Display(Name = "Albumin/Globulin Ratio")]
        [Range(0, 3, ErrorMessage = "A/G Ratio must be between 0 and 3")]
        public double AGRatio { get; set; } = 1.1;
    }

    /// <summary>
    /// View Model for disease pages with both input and result
    /// </summary>
    public class DiabetesViewModel
    {
        public DiabetesInput Input { get; set; } = new DiabetesInput();
        public PredictionResult? Result { get; set; }
    }

    public class HeartDiseaseViewModel
    {
        public HeartDiseaseInput Input { get; set; } = new HeartDiseaseInput();
        public PredictionResult? Result { get; set; }
    }

    public class ParkinsonsViewModel
    {
        public ParkinsonsInput Input { get; set; } = new ParkinsonsInput();
        public PredictionResult? Result { get; set; }
    }

    public class LiverViewModel
    {
        public LiverInput Input { get; set; } = new LiverInput();
        public PredictionResult? Result { get; set; }
    }

    /// <summary>
    /// Knee OA Prediction Result - specific for CNN image classification
    /// </summary>
    public class KneePredictionResult
    {
        public bool Success { get; set; }
        public int Prediction { get; set; }
        public string Grade { get; set; } = string.Empty;
        public string Classification { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string RiskLevel { get; set; } = string.Empty;
        public string Recommendation { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public Dictionary<string, double>? AllProbabilities { get; set; }
        public string Disease { get; set; } = string.Empty;
        public string? Error { get; set; }
        public string? Message { get; set; }
    }

    /// <summary>
    /// View Model for Knee OA Assessment
    /// </summary>
    public class KneeViewModel
    {
        public IFormFile? ImageFile { get; set; }
        public string? ImageBase64 { get; set; }
        public KneePredictionResult? Result { get; set; }
    }

    /// <summary>
    /// Error View Model
    /// </summary>
    public class ErrorViewModel
    {
        public string? RequestId { get; set; }
        public bool ShowRequestId => !string.IsNullOrEmpty(RequestId);
    }
}
