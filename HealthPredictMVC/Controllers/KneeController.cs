using Microsoft.AspNetCore.Mvc;
using HealthPredictMVC.Models;
using HealthPredictMVC.Services;

namespace HealthPredictMVC.Controllers
{
    public class KneeController : Controller
    {
        private readonly IPredictionService _predictionService;
        private readonly ILogger<KneeController> _logger;

        public KneeController(IPredictionService predictionService, ILogger<KneeController> logger)
        {
            _predictionService = predictionService;
            _logger = logger;
        }

        [HttpGet]
        public IActionResult Index()
        {
            var viewModel = new KneeViewModel();
            return View(viewModel);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Index(KneeViewModel viewModel)
        {
            try
            {
                if (viewModel.ImageFile == null || viewModel.ImageFile.Length == 0)
                {
                    ModelState.AddModelError(string.Empty, "Please upload a knee X-ray image.");
                    return View(viewModel);
                }

                // Validate file type
                var allowedTypes = new[] { "image/jpeg", "image/png", "image/jpg" };
                if (!allowedTypes.Contains(viewModel.ImageFile.ContentType.ToLower()))
                {
                    ModelState.AddModelError(string.Empty, "Please upload a valid image file (JPEG or PNG).");
                    return View(viewModel);
                }

                // Validate file size (max 10MB)
                if (viewModel.ImageFile.Length > 10 * 1024 * 1024)
                {
                    ModelState.AddModelError(string.Empty, "Image file size must be less than 10MB.");
                    return View(viewModel);
                }

                // Convert to base64 for display
                using var memoryStream = new MemoryStream();
                await viewModel.ImageFile.CopyToAsync(memoryStream);
                var imageBytes = memoryStream.ToArray();
                viewModel.ImageBase64 = $"data:{viewModel.ImageFile.ContentType};base64,{Convert.ToBase64String(imageBytes)}";

                // Get prediction
                var result = await _predictionService.PredictKneeAsync(viewModel.ImageFile);
                viewModel.Result = result;

                if (!result.Success)
                {
                    ModelState.AddModelError(string.Empty, result.Error ?? "An error occurred during prediction.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in knee OA prediction");
                ModelState.AddModelError(string.Empty, "An unexpected error occurred. Please try again.");
            }

            return View(viewModel);
        }
    }
}
