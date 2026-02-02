using System.Text;
using System.Text.Json;
using HealthPredictMVC.Models;

namespace HealthPredictMVC.Services
{
    public interface IPredictionService
    {
        Task<PredictionResult> PredictDiabetesAsync(DiabetesInput input);
        Task<PredictionResult> PredictHeartDiseaseAsync(HeartDiseaseInput input);
        Task<PredictionResult> PredictParkinsonsAsync(ParkinsonsInput input);
        Task<PredictionResult> PredictLiverAsync(LiverInput input);
        Task<KneePredictionResult> PredictKneeAsync(IFormFile imageFile);
        Task<KneePredictionResult> PredictKneeFromBase64Async(string base64Image);
        Task<bool> CheckHealthAsync();
    }

    public class PredictionService : IPredictionService
    {
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly ILogger<PredictionService> _logger;
        private readonly JsonSerializerOptions _jsonOptions;

        public PredictionService(IHttpClientFactory httpClientFactory, ILogger<PredictionService> logger)
        {
            _httpClientFactory = httpClientFactory;
            _logger = logger;
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
        }

        private HttpClient GetClient()
        {
            return _httpClientFactory.CreateClient("FlaskAPI");
        }

        public async Task<bool> CheckHealthAsync()
        {
            try
            {
                var client = GetClient();
                var response = await client.GetAsync("/api/health");
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Health check failed");
                return false;
            }
        }

        public async Task<PredictionResult> PredictDiabetesAsync(DiabetesInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "Pregnancies", input.Pregnancies },
                    { "Glucose", input.Glucose },
                    { "BloodPressure", input.BloodPressure },
                    { "SkinThickness", input.SkinThickness },
                    { "Insulin", input.Insulin },
                    { "BMI", input.BMI },
                    { "DiabetesPedigreeFunction", input.DiabetesPedigreeFunction },
                    { "Age", input.Age }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                _logger.LogInformation("Sending diabetes prediction request: {Json}", json);

                var response = await client.PostAsync("/api/diabetes/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                _logger.LogInformation("Received response: {Response}", responseContent);

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting diabetes");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<PredictionResult> PredictHeartDiseaseAsync(HeartDiseaseInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "Age", input.Age },
                    { "Sex", input.Sex },
                    { "Chest pain type", input.ChestPainType },
                    { "BP", input.BP },
                    { "Cholesterol", input.Cholesterol },
                    { "FBS over 120", input.FBSOver120 },
                    { "EKG results", input.EKGResults },
                    { "Max HR", input.MaxHR },
                    { "Exercise angina", input.ExerciseAngina },
                    { "ST depression", input.STDepression },
                    { "Slope of ST", input.SlopeOfST },
                    { "Number of vessels fluro", input.NumberOfVessels },
                    { "Thallium", input.Thallium }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PostAsync("/api/heart/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting heart disease");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<PredictionResult> PredictParkinsonsAsync(ParkinsonsInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "MDVP:Fo(Hz)", input.MDVP_Fo },
                    { "MDVP:Fhi(Hz)", input.MDVP_Fhi },
                    { "MDVP:Flo(Hz)", input.MDVP_Flo },
                    { "MDVP:Jitter(%)", input.MDVP_Jitter_Percent },
                    { "MDVP:Jitter(Abs)", input.MDVP_Jitter_Abs },
                    { "MDVP:RAP", input.MDVP_RAP },
                    { "MDVP:PPQ", input.MDVP_PPQ },
                    { "Jitter:DDP", input.Jitter_DDP },
                    { "MDVP:Shimmer", input.MDVP_Shimmer },
                    { "MDVP:Shimmer(dB)", input.MDVP_Shimmer_dB },
                    { "Shimmer:APQ3", input.Shimmer_APQ3 },
                    { "Shimmer:APQ5", input.Shimmer_APQ5 },
                    { "MDVP:APQ", input.MDVP_APQ },
                    { "Shimmer:DDA", input.Shimmer_DDA },
                    { "NHR", input.NHR },
                    { "HNR", input.HNR },
                    { "RPDE", input.RPDE },
                    { "DFA", input.DFA },
                    { "spread1", input.Spread1 },
                    { "spread2", input.Spread2 },
                    { "D2", input.D2 },
                    { "PPE", input.PPE }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PostAsync("/api/parkinsons/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting Parkinson's");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<PredictionResult> PredictLiverAsync(LiverInput input)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, object>
                {
                    { "Age", input.Age },
                    { "Gender", input.Gender },
                    { "Total_Bilirubin", input.TotalBilirubin },
                    { "Direct_Bilirubin", input.DirectBilirubin },
                    { "Alkaline_Phosphotase", input.AlkalinePhosphatase },
                    { "Alamine_Aminotransferase", input.ALT },
                    { "Aspartate_Aminotransferase", input.AST },
                    { "Total_Protiens", input.TotalProteins },
                    { "Albumin", input.Albumin },
                    { "Albumin_and_Globulin_Ratio", input.AGRatio }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PostAsync("/api/liver/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<PredictionResult>(responseContent, _jsonOptions);
                    return result ?? new PredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    return new PredictionResult
                    {
                        Success = false,
                        Error = $"API returned status code: {response.StatusCode}"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting liver disease");
                return new PredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }

        public async Task<KneePredictionResult> PredictKneeAsync(IFormFile imageFile)
        {
            try
            {
                var client = GetClient();

                using var memoryStream = new MemoryStream();
                await imageFile.CopyToAsync(memoryStream);
                var imageBytes = memoryStream.ToArray();
                var base64Image = Convert.ToBase64String(imageBytes);

                return await PredictKneeFromBase64Async(base64Image);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting knee OA");
                return new KneePredictionResult
                {
                    Success = false,
                    Error = $"Error processing image: {ex.Message}"
                };
            }
        }

        public async Task<KneePredictionResult> PredictKneeFromBase64Async(string base64Image)
        {
            try
            {
                var client = GetClient();

                var requestData = new Dictionary<string, string>
                {
                    { "image", base64Image }
                };

                var json = JsonSerializer.Serialize(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                _logger.LogInformation("Sending knee OA prediction request");

                var response = await client.PostAsync("/api/knee/predict", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                _logger.LogInformation("Received knee response: {Response}", responseContent.Substring(0, Math.Min(200, responseContent.Length)));

                if (response.IsSuccessStatusCode)
                {
                    var result = JsonSerializer.Deserialize<KneePredictionResult>(responseContent, _jsonOptions);
                    return result ?? new KneePredictionResult { Success = false, Error = "Failed to parse response" };
                }
                else
                {
                    // Try to parse error message from response
                    try
                    {
                        var errorResult = JsonSerializer.Deserialize<KneePredictionResult>(responseContent, _jsonOptions);
                        return errorResult ?? new KneePredictionResult
                        {
                            Success = false,
                            Error = $"API returned status code: {response.StatusCode}"
                        };
                    }
                    catch
                    {
                        return new KneePredictionResult
                        {
                            Success = false,
                            Error = $"API returned status code: {response.StatusCode}"
                        };
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error predicting knee OA");
                return new KneePredictionResult
                {
                    Success = false,
                    Error = $"Error connecting to prediction service: {ex.Message}"
                };
            }
        }
    }
}
