"""
Advanced Analytics API Routes for Phase 7.5 Integration.

This module provides FastAPI routes for advanced analytics features including:
- Advanced Machine Learning Models
- Enhanced Predictive Analytics
- Real-Time Analytics Dashboard
- Advanced Data Processing
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import timedelta
import asyncio
import logging
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/advanced-analytics", tags=["Advanced Analytics"])

# Initialize components conditionally to avoid import errors
forecasting_engine = None
causal_engine = None
scenario_engine = None
confidence_calculator = None
anomaly_detector = None
model_optimizer = None
feature_engineer = None
performance_monitor = None

dl_engine = None
transfer_service = None
model_versioning = None
automl_pipeline = None
ensemble_methods = None
time_series_models = None
clustering_algorithms = None
dimensionality_reduction = None

forecasting_agent = None
causal_agent = None
anomaly_agent = None
ml_agent = None

# Try to import and initialize components
try:
    from src.core.advanced_analytics.multivariate_forecasting import MultivariateForecastingEngine
    forecasting_engine = MultivariateForecastingEngine()
    logger.info("✅ MultivariateForecastingEngine initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize MultivariateForecastingEngine: {e}")

try:
    from src.core.advanced_analytics.causal_inference_engine import CausalInferenceEngine
    causal_engine = CausalInferenceEngine()
    logger.info("✅ CausalInferenceEngine initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize CausalInferenceEngine: {e}")

try:
    from src.core.advanced_analytics.scenario_analysis import ScenarioAnalysisEngine
    scenario_engine = ScenarioAnalysisEngine()
    logger.info("✅ ScenarioAnalysisEngine initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize ScenarioAnalysisEngine: {e}")

try:
    from src.core.advanced_analytics.confidence_intervals import ConfidenceIntervalCalculator
    confidence_calculator = ConfidenceIntervalCalculator()
    logger.info("✅ ConfidenceIntervalCalculator initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize ConfidenceIntervalCalculator: {e}")

try:
    from src.core.advanced_analytics.advanced_anomaly_detection import AdvancedAnomalyDetector
    anomaly_detector = AdvancedAnomalyDetector()
    logger.info("✅ AdvancedAnomalyDetector initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize AdvancedAnomalyDetector: {e}")

try:
    from src.core.advanced_analytics.model_optimization import ModelOptimizer
    model_optimizer = ModelOptimizer()
    logger.info("✅ ModelOptimizer initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize ModelOptimizer: {e}")

try:
    from src.core.advanced_analytics.feature_engineering import FeatureEngineer
    feature_engineer = FeatureEngineer()
    logger.info("✅ FeatureEngineer initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize FeatureEngineer: {e}")

try:
    from src.core.advanced_analytics.performance_monitoring import AdvancedPerformanceMonitor
    performance_monitor = AdvancedPerformanceMonitor()
    logger.info("✅ AdvancedPerformanceMonitor initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize AdvancedPerformanceMonitor: {e}")

# Try to import ML components
try:
    from src.core.advanced_ml.deep_learning_engine import DeepLearningEngine
    dl_engine = DeepLearningEngine()
    logger.info("✅ DeepLearningEngine initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize DeepLearningEngine: {e}")

try:
    from src.core.advanced_ml.transfer_learning_service import TransferLearningService
    transfer_service = TransferLearningService()
    logger.info("✅ TransferLearningService initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize TransferLearningService: {e}")

try:
    from src.core.advanced_ml.model_versioning import ModelVersioning
    model_versioning = ModelVersioning()
    logger.info("✅ ModelVersioning initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize ModelVersioning: {e}")

try:
    from src.core.advanced_ml.automl_pipeline import AutoMLPipeline
    automl_pipeline = AutoMLPipeline()
    logger.info("✅ AutoMLPipeline initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize AutoMLPipeline: {e}")

try:
    from src.core.advanced_ml.ensemble_methods import EnsembleMethods
    ensemble_methods = EnsembleMethods()
    logger.info("✅ EnsembleMethods initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize EnsembleMethods: {e}")

try:
    from src.core.advanced_ml.time_series_models import TimeSeriesModels
    time_series_models = TimeSeriesModels()
    logger.info("✅ TimeSeriesModels initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize TimeSeriesModels: {e}")

try:
    from src.core.advanced_ml.clustering_algorithms import ClusteringAlgorithms
    clustering_algorithms = ClusteringAlgorithms()
    logger.info("✅ ClusteringAlgorithms initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize ClusteringAlgorithms: {e}")

try:
    from src.core.advanced_ml.dimensionality_reduction import DimensionalityReduction
    dimensionality_reduction = DimensionalityReduction()
    logger.info("✅ DimensionalityReduction initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize DimensionalityReduction: {e}")

# Try to import agents
try:
    from src.agents.advanced_forecasting_agent import AdvancedForecastingAgent
    forecasting_agent = AdvancedForecastingAgent()
    logger.info("✅ AdvancedForecastingAgent initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize AdvancedForecastingAgent: {e}")

try:
    from src.agents.causal_analysis_agent import CausalAnalysisAgent
    causal_agent = CausalAnalysisAgent()
    logger.info("✅ CausalAnalysisAgent initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize CausalAnalysisAgent: {e}")

try:
    from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
    anomaly_agent = AnomalyDetectionAgent()
    logger.info("✅ AnomalyDetectionAgent initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize AnomalyDetectionAgent: {e}")

try:
    from src.agents.advanced_ml_agent import AdvancedMLAgent
    ml_agent = AdvancedMLAgent()
    logger.info("✅ AdvancedMLAgent initialized")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize AdvancedMLAgent: {e}")

# Import error handling
try:
    from src.core.error_handler import with_error_handling
except Exception as e:
    logger.warning(f"⚠️ Could not import error handler: {e}")
    # Create a simple fallback decorator that takes operation_name parameter
    def with_error_handling(operation_name: str = None):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {operation_name or func.__name__}: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            return wrapper
        return decorator

# Pydantic models for request/response
class ForecastingRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_variables: List[str]
    forecast_horizon: int
    model_type: str = "ensemble"
    confidence_level: float = 0.95

class CausalAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    variables: List[str]
    analysis_type: str = "granger"
    max_lag: int = 5

class ScenarioAnalysisRequest(BaseModel):
    base_data: List[Dict[str, Any]]
    scenarios: List[Dict[str, Any]]
    target_variable: str
    analysis_type: str = "impact"

class AnomalyDetectionRequest(BaseModel):
    data: List[Dict[str, Any]]
    algorithm: str = "isolation_forest"
    threshold: float = 0.1
    features: Optional[List[str]] = None

class ModelOptimizationRequest(BaseModel):
    config: Dict[str, Any]
    optimization_type: str = "hyperparameter"
    metric: str = "accuracy"

class FeatureEngineeringRequest(BaseModel):
    data: List[Dict[str, Any]]
    features: List[str]
    engineering_type: str = "automatic"

class PerformanceMonitoringRequest(BaseModel):
    metrics: List[str]
    time_range: str = "1h"
    aggregation: str = "mean"

class DeepLearningRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_type: str = "mlp"
    task: str = "classification"
    config: Optional[Dict[str, Any]] = None

class TransferLearningRequest(BaseModel):
    source_data: List[Dict[str, Any]]
    target_data: List[Dict[str, Any]]
    model_type: str = "pretrained"
    fine_tune: bool = True

class AutoMLRequest(BaseModel):
    data: List[Dict[str, Any]]
    target: str
    task: str = "classification"
    time_limit: int = 3600

class EnsembleRequest(BaseModel):
    data: List[Dict[str, Any]]
    target: str
    methods: List[str] = ["random_forest", "xgboost", "lightgbm"]
    voting: str = "soft"

class TimeSeriesRequest(BaseModel):
    data: List[Dict[str, Any]]
    target: str
    model_type: str = "arima"
    forecast_steps: int = 10

class ClusteringRequest(BaseModel):
    data: List[Dict[str, Any]]
    algorithm: str = "kmeans"
    n_clusters: int = 3
    features: Optional[List[str]] = None

class DimensionalityReductionRequest(BaseModel):
    data: List[Dict[str, Any]]
    method: str = "pca"
    n_components: int = 2
    features: Optional[List[str]] = None

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for advanced analytics services."""
    components_status = {
        "forecasting_engine": forecasting_engine is not None,
        "causal_engine": causal_engine is not None,
        "scenario_engine": scenario_engine is not None,
        "confidence_calculator": confidence_calculator is not None,
        "anomaly_detector": anomaly_detector is not None,
        "model_optimizer": model_optimizer is not None,
        "feature_engineer": feature_engineer is not None,
        "performance_monitor": performance_monitor is not None,
        "dl_engine": dl_engine is not None,
        "transfer_service": transfer_service is not None,
        "model_versioning": model_versioning is not None,
        "automl_pipeline": automl_pipeline is not None,
        "ensemble_methods": ensemble_methods is not None,
        "time_series_models": time_series_models is not None,
        "clustering_algorithms": clustering_algorithms is not None,
        "dimensionality_reduction": dimensionality_reduction is not None,
        "forecasting_agent": forecasting_agent is not None,
        "causal_agent": causal_agent is not None,
        "anomaly_agent": anomaly_agent is not None,
        "ml_agent": ml_agent is not None
    }
    
    available_components = sum(components_status.values())
    total_components = len(components_status)
    
    return {
        "status": "healthy" if available_components > 0 else "degraded",
        "available_components": available_components,
        "total_components": total_components,
        "components_status": components_status
    }

# Forecasting endpoints
@router.post("/forecasting-test")
async def multivariate_forecasting_test(request: ForecastingRequest):
    """Perform multivariate time series forecasting."""
    try:
        # For now, return a mock response to test the endpoint
        return JSONResponse(content={
            "success": True, 
            "result": {
                "forecast": [100, 105, 110, 115, 120, 125, 130],
                "confidence_intervals": {"sales": [[95, 105], [100, 110], [105, 115], [110, 120], [115, 125], [120, 130], [125, 135]]},
                "model_performance": {"mae": 2.5, "rmse": 3.1},
                "forecast_horizon": request.forecast_horizon,
                "target_variables": request.target_variables
            }
        })
    except Exception as e:
        logger.error(f"Error in multivariate forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecasting/agent")
@with_error_handling
async def agent_forecasting(request: ForecastingRequest):
    """Perform forecasting using the advanced forecasting agent."""
    if forecasting_agent is None:
        raise HTTPException(status_code=503, detail="Forecasting agent not available")
    
    try:
        result = await forecasting_agent.forecast(
            data=request.data,
            target_variables=request.target_variables,
            forecast_horizon=request.forecast_horizon,
            model_type=request.model_type
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in agent forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Causal analysis endpoints
@router.post("/causal/analysis")
@with_error_handling
async def causal_analysis(request: CausalAnalysisRequest):
    """Perform causal inference analysis."""
    if causal_engine is None:
        raise HTTPException(status_code=503, detail="Causal engine not available")
    
    try:
        result = await causal_engine.analyze_causality(
            data=request.data,
            variables=request.variables,
            analysis_type=request.analysis_type,
            max_lag=request.max_lag
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in causal analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/causal/agent")
@with_error_handling
async def agent_causal_analysis(request: CausalAnalysisRequest):
    """Perform causal analysis using the causal analysis agent."""
    if causal_agent is None:
        raise HTTPException(status_code=503, detail="Causal agent not available")
    
    try:
        result = await causal_agent.analyze_causality(
            data=request.data,
            variables=request.variables,
            analysis_type=request.analysis_type
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in agent causal analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scenario analysis endpoints
@router.post("/scenario/analysis")
@with_error_handling
async def scenario_analysis(request: ScenarioAnalysisRequest):
    """Perform scenario analysis."""
    if scenario_engine is None:
        raise HTTPException(status_code=503, detail="Scenario engine not available")
    
    try:
        result = await scenario_engine.analyze_scenarios(
            base_data=request.base_data,
            scenarios=request.scenarios,
            target_variable=request.target_variable,
            analysis_type=request.analysis_type
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in scenario analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Anomaly detection endpoints
@router.post("/anomaly/detection")
@with_error_handling
async def anomaly_detection(request: AnomalyDetectionRequest):
    """Perform advanced anomaly detection."""
    if anomaly_detector is None:
        raise HTTPException(status_code=503, detail="Anomaly detector not available")
    
    try:
        result = await anomaly_detector.detect_anomalies(
            data=request.data,
            algorithm=request.algorithm,
            threshold=request.threshold,
            features=request.features
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/anomaly/agent")
@with_error_handling
async def agent_anomaly_detection(request: AnomalyDetectionRequest):
    """Perform anomaly detection using the anomaly detection agent."""
    if anomaly_agent is None:
        raise HTTPException(status_code=503, detail="Anomaly agent not available")
    
    try:
        result = await anomaly_agent.detect_anomalies(
            data=request.data,
            algorithm=request.algorithm,
            threshold=request.threshold
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in agent anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model optimization endpoints
@router.post("/optimization/model")
@with_error_handling
async def model_optimization(request: ModelOptimizationRequest):
    """Optimize machine learning models."""
    if model_optimizer is None:
        raise HTTPException(status_code=503, detail="Model optimizer not available")
    
    try:
        result = await model_optimizer.optimize_model(
            model_config=request.config,
            optimization_type=request.optimization_type,
            metric=request.metric
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in model optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feature engineering endpoints
@router.post("/features/engineering")
@with_error_handling
async def feature_engineering(request: FeatureEngineeringRequest):
    """Perform automated feature engineering."""
    if feature_engineer is None:
        raise HTTPException(status_code=503, detail="Feature engineer not available")
    
    try:
        result = await feature_engineer.engineer_features(
            data=request.data,
            features=request.features,
            engineering_type=request.engineering_type
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance monitoring endpoints
@router.post("/monitoring/performance")
async def performance_monitoring(request: PerformanceMonitoringRequest):
    """Monitor system performance metrics."""
    if performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitor not available")
    
    try:
        # Convert time_range string to timedelta
        time_window = None
        if request.time_range == "1h":
            time_window = timedelta(hours=1)
        elif request.time_range == "24h":
            time_window = timedelta(hours=24)
        elif request.time_range == "7d":
            time_window = timedelta(days=7)
        
        # Get performance summary using the correct method
        result = performance_monitor.get_performance_summary(
            metric_names=request.metrics,
            time_window=time_window
        )
        
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in performance monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Deep learning endpoints
@router.post("/ml/deep-learning")
@with_error_handling
async def deep_learning(request: DeepLearningRequest):
    """Train and use deep learning models."""
    if dl_engine is None:
        raise HTTPException(status_code=503, detail="Deep learning engine not available")
    
    try:
        result = await dl_engine.create_and_train_model(
            data=request.data,
            model_type=request.model_type,
            task=request.task,
            config=request.config
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in deep learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Transfer learning endpoints
@router.post("/ml/transfer-learning")
@with_error_handling
async def transfer_learning(request: TransferLearningRequest):
    """Perform transfer learning."""
    if transfer_service is None:
        raise HTTPException(status_code=503, detail="Transfer learning service not available")
    
    try:
        result = await transfer_service.transfer_learn(
            source_data=request.source_data,
            target_data=request.target_data,
            model_type=request.model_type,
            fine_tune=request.fine_tune
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in transfer learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AutoML endpoints
@router.post("/ml/automl")
@with_error_handling
async def automl_pipeline_endpoint(request: AutoMLRequest):
    """Run AutoML pipeline."""
    if automl_pipeline is None:
        raise HTTPException(status_code=503, detail="AutoML pipeline not available")
    
    try:
        result = await automl_pipeline.run_pipeline(
            data=request.data,
            target=request.target,
            task=request.task,
            time_limit=request.time_limit
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in AutoML pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ensemble methods endpoints
@router.post("/ml/ensemble")
@with_error_handling
async def ensemble_methods_endpoint(request: EnsembleRequest):
    """Use ensemble methods."""
    if ensemble_methods is None:
        raise HTTPException(status_code=503, detail="Ensemble methods not available")
    
    try:
        result = await ensemble_methods.create_ensemble(
            data=request.data,
            target=request.target,
            methods=request.methods,
            voting=request.voting
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in ensemble methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Time series models endpoints
@router.post("/ml/time-series")
@with_error_handling
async def time_series_models_endpoint(request: TimeSeriesRequest):
    """Use time series models."""
    if time_series_models is None:
        raise HTTPException(status_code=503, detail="Time series models not available")
    
    try:
        result = await time_series_models.train_model(
            data=request.data,
            target=request.target,
            model_type=request.model_type,
            forecast_steps=request.forecast_steps
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in time series models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Clustering algorithms endpoints
@router.post("/ml/clustering")
@with_error_handling
async def clustering_algorithms_endpoint(request: ClusteringRequest):
    """Use clustering algorithms."""
    if clustering_algorithms is None:
        raise HTTPException(status_code=503, detail="Clustering algorithms not available")
    
    try:
        result = await clustering_algorithms.cluster_data(
            data=request.data,
            algorithm=request.algorithm,
            n_clusters=request.n_clusters,
            features=request.features
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in clustering algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dimensionality reduction endpoints
@router.post("/ml/dimensionality-reduction")
@with_error_handling
async def dimensionality_reduction_endpoint(request: DimensionalityReductionRequest):
    """Perform dimensionality reduction."""
    if dimensionality_reduction is None:
        raise HTTPException(status_code=503, detail="Dimensionality reduction not available")
    
    try:
        result = await dimensionality_reduction.reduce_dimensions(
            data=request.data,
            method=request.method,
            n_components=request.n_components,
            features=request.features
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced ML agent endpoints
@router.post("/ml/agent")
@with_error_handling
async def advanced_ml_agent_endpoint(request: DeepLearningRequest):
    """Use the advanced ML agent."""
    if ml_agent is None:
        raise HTTPException(status_code=503, detail="Advanced ML agent not available")
    
    try:
        result = await ml_agent.process_ml_request(
            data=request.data,
            model_type=request.model_type,
            task=request.task,
            config=request.config
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in advanced ML agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model versioning endpoints
@router.post("/ml/versioning")
@with_error_handling
async def model_versioning_endpoint(request: Dict[str, Any]):
    """Manage model versions."""
    if model_versioning is None:
        raise HTTPException(status_code=503, detail="Model versioning not available")
    
    try:
        action = request.get("action", "save")
        if action == "save":
            result = await model_versioning.save_model_version(request)
        elif action == "load":
            result = await model_versioning.load_model_version(request)
        elif action == "list":
            result = await model_versioning.list_model_versions(request)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
        
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in model versioning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Confidence intervals endpoints
@router.post("/confidence/intervals")
@with_error_handling
async def confidence_intervals_endpoint(request: Dict[str, Any]):
    """Calculate confidence intervals."""
    if confidence_calculator is None:
        raise HTTPException(status_code=503, detail="Confidence calculator not available")
    
    try:
        result = await confidence_calculator.calculate_confidence_intervals(
            data=request.get("data", []),
            confidence_level=request.get("confidence_level", 0.95),
            method=request.get("method", "bootstrap")
        )
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        logger.error(f"Error in confidence intervals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

logger.info("✅ Advanced analytics routes initialized with conditional component loading")
