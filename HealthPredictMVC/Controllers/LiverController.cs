using Microsoft.AspNetCore.Mvc;
using HealthPredictMVC.Models;
using HealthPredictMVC.Services;

namespace HealthPredictMVC.Controllers
{
    public class LiverController : Controller
    {
        private readonly IPredictionService _predictionService;
        private readonly ILogger<LiverController> _logger;

        public LiverController(IPredictionService predictionService, ILogger<LiverController> logger)
        {
            _predictionService = predictionService;
            _logger = logger;
        }

        [HttpGet]
        public IActionResult Index()
        {
            var viewModel = new LiverViewModel();
            return View(viewModel);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Index(LiverViewModel viewModel)
        {
            if (!ModelState.IsValid)
            {
                return View(viewModel);
            }

            try
            {
                var result = await _predictionService.PredictLiverAsync(viewModel.Input);
                viewModel.Result = result;

                if (!result.Success)
                {
                    ModelState.AddModelError(string.Empty, result.Error ?? "An error occurred during prediction.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in liver disease prediction");
                ModelState.AddModelError(string.Empty, "An unexpected error occurred. Please try again.");
            }

            return View(viewModel);
        }
    }
}
