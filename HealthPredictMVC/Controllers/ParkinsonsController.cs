using Microsoft.AspNetCore.Mvc;
using HealthPredictMVC.Models;
using HealthPredictMVC.Services;

namespace HealthPredictMVC.Controllers
{
    public class ParkinsonsController : Controller
    {
        private readonly IPredictionService _predictionService;
        private readonly ILogger<ParkinsonsController> _logger;

        public ParkinsonsController(IPredictionService predictionService, ILogger<ParkinsonsController> logger)
        {
            _predictionService = predictionService;
            _logger = logger;
        }

        [HttpGet]
        public IActionResult Index()
        {
            var viewModel = new ParkinsonsViewModel();
            return View(viewModel);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Index(ParkinsonsViewModel viewModel)
        {
            if (!ModelState.IsValid)
            {
                return View(viewModel);
            }

            try
            {
                var result = await _predictionService.PredictParkinsonsAsync(viewModel.Input);
                viewModel.Result = result;

                if (!result.Success)
                {
                    ModelState.AddModelError(string.Empty, result.Error ?? "An error occurred during prediction.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in Parkinson's prediction");
                ModelState.AddModelError(string.Empty, "An unexpected error occurred. Please try again.");
            }

            return View(viewModel);
        }
    }
}
