using Microsoft.AspNetCore.Mvc;
using HealthPredictMVC.Models;
using HealthPredictMVC.Services;

namespace HealthPredictMVC.Controllers
{
    public class DiabetesController : Controller
    {
        private readonly IPredictionService _predictionService;
        private readonly ILogger<DiabetesController> _logger;

        public DiabetesController(IPredictionService predictionService, ILogger<DiabetesController> logger)
        {
            _predictionService = predictionService;
            _logger = logger;
        }

        [HttpGet]
        public IActionResult Index()
        {
            var viewModel = new DiabetesViewModel();
            return View(viewModel);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Index(DiabetesViewModel viewModel)
        {
            if (!ModelState.IsValid)
            {
                return View(viewModel);
            }

            try
            {
                var result = await _predictionService.PredictDiabetesAsync(viewModel.Input);
                viewModel.Result = result;

                if (!result.Success)
                {
                    ModelState.AddModelError(string.Empty, result.Error ?? "An error occurred during prediction.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in diabetes prediction");
                ModelState.AddModelError(string.Empty, "An unexpected error occurred. Please try again.");
            }

            return View(viewModel);
        }
    }
}
