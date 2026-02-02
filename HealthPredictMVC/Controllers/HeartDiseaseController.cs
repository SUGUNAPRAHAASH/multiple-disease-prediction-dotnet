using Microsoft.AspNetCore.Mvc;
using HealthPredictMVC.Models;
using HealthPredictMVC.Services;

namespace HealthPredictMVC.Controllers
{
    public class HeartDiseaseController : Controller
    {
        private readonly IPredictionService _predictionService;
        private readonly ILogger<HeartDiseaseController> _logger;

        public HeartDiseaseController(IPredictionService predictionService, ILogger<HeartDiseaseController> logger)
        {
            _predictionService = predictionService;
            _logger = logger;
        }

        [HttpGet]
        public IActionResult Index()
        {
            var viewModel = new HeartDiseaseViewModel();
            return View(viewModel);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Index(HeartDiseaseViewModel viewModel)
        {
            if (!ModelState.IsValid)
            {
                return View(viewModel);
            }

            try
            {
                var result = await _predictionService.PredictHeartDiseaseAsync(viewModel.Input);
                viewModel.Result = result;

                if (!result.Success)
                {
                    ModelState.AddModelError(string.Empty, result.Error ?? "An error occurred during prediction.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in heart disease prediction");
                ModelState.AddModelError(string.Empty, "An unexpected error occurred. Please try again.");
            }

            return View(viewModel);
        }
    }
}
