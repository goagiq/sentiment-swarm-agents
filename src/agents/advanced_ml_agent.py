"""
Advanced Machine Learning Agent

This agent provides advanced machine learning capabilities including deep learning,
transfer learning, ensemble methods, and automated ML pipelines.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.core.models import AnalysisRequest, AnalysisResult, ProcessingStatus
from src.core.advanced_ml.deep_learning_engine import DeepLearningEngine
from src.core.advanced_ml.transfer_learning_service import TransferLearningService
from src.core.advanced_ml.model_versioning import ModelVersioning
from src.core.advanced_ml.ensemble_methods import EnsembleMethods
from src.core.advanced_ml.time_series_models import TimeSeriesModels
from src.core.advanced_ml.clustering_algorithms import ClusteringAlgorithms
from src.core.advanced_ml.dimensionality_reduction import DimensionalityReduction
from src.core.advanced_ml.automl_pipeline import AutoMLPipeline
from src.config.advanced_ml_config import get_advanced_ml_config

logger = logging.getLogger(__name__)


class AdvancedMLAgent(BaseAgent):
    """Advanced Machine Learning Agent for sophisticated ML operations."""
    
    def __init__(self):
        super().__init__()
        self.config = get_advanced_ml_config()
        self.agent_name = "AdvancedMLAgent"
        self.agent_description = "Advanced machine learning capabilities including deep learning, transfer learning, and automated ML"
        
        # Initialize ML components
        self.deep_learning_engine = DeepLearningEngine()
        self.transfer_learning_service = TransferLearningService()
        self.model_versioning = ModelVersioning()
        self.ensemble_methods = EnsembleMethods()
        self.time_series_models = TimeSeriesModels()
        self.clustering_algorithms = ClusteringAlgorithms()
        self.dimensionality_reduction = DimensionalityReduction()
        self.automl_pipeline = AutoMLPipeline()
        
        # Agent capabilities
        self.capabilities = {
            "deep_learning": {
                "description": "Create and train deep learning models",
                "methods": ["create_mlp", "create_cnn", "create_lstm", "train_model", "predict"]
            },
            "transfer_learning": {
                "description": "Load pre-trained models and fine-tune them",
                "methods": ["load_pre_trained_model", "create_fine_tuned_model", "fine_tune_model"]
            },
            "model_versioning": {
                "description": "Manage model versions and lifecycle",
                "methods": ["create_version", "load_version", "list_versions", "compare_versions"]
            },
            "ensemble_methods": {
                "description": "Create and train ensemble models",
                "methods": ["create_ensemble", "train_ensemble", "predict_ensemble"]
            },
            "time_series": {
                "description": "Time series forecasting and analysis",
                "methods": ["create_time_series_model", "forecast", "analyze_seasonality"]
            },
            "clustering": {
                "description": "Clustering and unsupervised learning",
                "methods": ["create_clustering_model", "cluster_data", "analyze_clusters"]
            },
            "dimensionality_reduction": {
                "description": "Reduce data dimensionality",
                "methods": ["reduce_dimensions", "visualize_reduced_data"]
            },
            "automl": {
                "description": "Automated machine learning pipeline",
                "methods": ["run_automl", "get_best_model", "optimize_hyperparameters"]
            }
        }
        
        logger.info(f"Initialized {self.agent_name}")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        try:
            # Check if request contains ML-related data
            content = request.content.lower()
            keywords = ['ml', 'machine learning', 'deep learning', 'neural network', 'model', 'train', 'predict']
            return any(keyword in content for keyword in keywords)
        except Exception as e:
            logger.error(f"Error checking if agent can process request: {e}")
            return False
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        try:
            # Convert AnalysisRequest to dict format for processing
            request_dict = {
                'content': request.content,
                'type': 'deep_learning',  # Default type
                'metadata': request.metadata or {}
            }
            
            # Process the request
            result = self.process_request(request_dict)
            
            # Convert result to AnalysisResult
            return AnalysisResult(
                id=request.id,
                content=request.content,
                result=result,
                status=ProcessingStatus.COMPLETED,
                processing_time=0.0,
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return AnalysisResult(
                id=request.id,
                content=request.content,
                result={'error': str(e)},
                status=ProcessingStatus.FAILED,
                processing_time=0.0,
                metadata={}
            )
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an advanced ML request."""
        try:
            request_type = request.get("type", "")
            operation = request.get("operation", "")
            
            logger.info(f"Processing {request_type} operation: {operation}")
            
            if request_type == "deep_learning":
                return self._handle_deep_learning_request(request)
            elif request_type == "transfer_learning":
                return self._handle_transfer_learning_request(request)
            elif request_type == "model_versioning":
                return self._handle_model_versioning_request(request)
            elif request_type == "ensemble_methods":
                return self._handle_ensemble_request(request)
            elif request_type == "time_series":
                return self._handle_time_series_request(request)
            elif request_type == "clustering":
                return self._handle_clustering_request(request)
            elif request_type == "dimensionality_reduction":
                return self._handle_dimensionality_reduction_request(request)
            elif request_type == "automl":
                return self._handle_automl_request(request)
            else:
                return self._create_error_response(f"Unknown request type: {request_type}")
                
        except Exception as e:
            return self._create_error_response(f"Error processing request: {str(e)}")
    
    def _handle_deep_learning_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle deep learning requests."""
        try:
            operation = request.get("operation", "")
            data = request.get("data", {})
            
            if operation == "create_mlp":
                input_dim = data.get("input_dim")
                output_dim = data.get("output_dim")
                architecture = data.get("architecture", "mlp")
                
                model = self.deep_learning_engine.create_mlp(input_dim, output_dim, architecture)
                return self._create_success_response("MLP model created", {"model": str(model)})
            
            elif operation == "create_cnn":
                input_shape = tuple(data.get("input_shape", []))
                output_dim = data.get("output_dim")
                
                model = self.deep_learning_engine.create_cnn(input_shape, output_dim)
                return self._create_success_response("CNN model created", {"model": str(model)})
            
            elif operation == "create_lstm":
                input_shape = tuple(data.get("input_shape", []))
                output_dim = data.get("output_dim")
                
                model = self.deep_learning_engine.create_lstm(input_shape, output_dim)
                return self._create_success_response("LSTM model created", {"model": str(model)})
            
            elif operation == "train_model":
                model = data.get("model")
                X_train = np.array(data.get("X_train", []))
                y_train = np.array(data.get("y_train", []))
                X_val = np.array(data.get("X_val", [])) if data.get("X_val") else None
                y_val = np.array(data.get("y_val", [])) if data.get("y_val") else None
                model_name = data.get("model_name", "model")
                
                result = self.deep_learning_engine.train_model(
                    model, X_train, y_train, X_val, y_val, model_name
                )
                return self._create_success_response("Model trained", result)
            
            elif operation == "predict":
                model = data.get("model")
                X = np.array(data.get("X", []))
                
                predictions = self.deep_learning_engine.predict(model, X)
                return self._create_success_response("Predictions made", {"predictions": predictions.tolist()})
            
            else:
                return self._create_error_response(f"Unknown deep learning operation: {operation}")
                
        except Exception as e:
            return self._create_error_response(f"Error in deep learning operation: {str(e)}")
    
    def _handle_transfer_learning_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transfer learning requests."""
        try:
            operation = request.get("operation", "")
            data = request.get("data", {})
            
            if operation == "load_pre_trained_model":
                model_type = data.get("model_type")
                model_name = data.get("model_name")
                
                model = self.transfer_learning_service.load_pre_trained_model(model_type, model_name)
                return self._create_success_response("Pre-trained model loaded", {"model": str(model)})
            
            elif operation == "create_fine_tuned_model":
                base_model_name = data.get("base_model_name")
                output_dim = data.get("output_dim")
                task_type = data.get("task_type", "classification")
                
                model = self.transfer_learning_service.create_fine_tuned_model(
                    base_model_name, output_dim, task_type
                )
                return self._create_success_response("Fine-tuned model created", {"model": str(model)})
            
            elif operation == "fine_tune_model":
                fine_tuned_model = data.get("fine_tuned_model")
                X_train = np.array(data.get("X_train", []))
                y_train = np.array(data.get("y_train", []))
                X_val = np.array(data.get("X_val", [])) if data.get("X_val") else None
                y_val = np.array(data.get("y_val", [])) if data.get("y_val") else None
                model_name = data.get("model_name", "fine_tuned")
                
                result = self.transfer_learning_service.fine_tune_model(
                    fine_tuned_model, X_train, y_train, X_val, y_val, model_name
                )
                return self._create_success_response("Model fine-tuned", result)
            
            else:
                return self._create_error_response(f"Unknown transfer learning operation: {operation}")
                
        except Exception as e:
            return self._create_error_response(f"Error in transfer learning operation: {str(e)}")
    
    def _handle_model_versioning_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model versioning requests."""
        try:
            operation = request.get("operation", "")
            data = request.get("data", {})
            
            if operation == "create_version":
                model_name = data.get("model_name")
                model = data.get("model")
                metadata = data.get("metadata", {})
                version_format = data.get("version_format", "semantic")
                
                version = self.model_versioning.create_version(
                    model_name, model, metadata, version_format
                )
                return self._create_success_response("Version created", {"version": version})
            
            elif operation == "load_version":
                model_name = data.get("model_name")
                version = data.get("version")
                
                model = self.model_versioning.load_version(model_name, version)
                return self._create_success_response("Version loaded", {"model": str(model)})
            
            elif operation == "list_versions":
                model_name = data.get("model_name")
                
                versions = self.model_versioning.list_versions(model_name)
                return self._create_success_response("Versions listed", {"versions": versions})
            
            elif operation == "compare_versions":
                model_name = data.get("model_name")
                version1 = data.get("version1")
                version2 = data.get("version2")
                
                comparison = self.model_versioning.compare_versions(model_name, version1, version2)
                return self._create_success_response("Versions compared", comparison)
            
            else:
                return self._create_error_response(f"Unknown model versioning operation: {operation}")
                
        except Exception as e:
            return self._create_error_response(f"Error in model versioning operation: {str(e)}")
    
    def _handle_ensemble_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ensemble methods requests."""
        try:
            operation = request.get("operation", "")
            data = request.get("data", {})
            
            if operation == "create_ensemble":
                ensemble_type = data.get("ensemble_type")
                base_models = data.get("base_models", [])
                config = data.get("config", {})
                
                ensemble = self.ensemble_methods.create_ensemble(ensemble_type, base_models, config)
                return self._create_success_response("Ensemble created", {"ensemble": str(ensemble)})
            
            elif operation == "train_ensemble":
                ensemble = data.get("ensemble")
                X_train = np.array(data.get("X_train", []))
                y_train = np.array(data.get("y_train", []))
                
                result = self.ensemble_methods.train_ensemble(ensemble, X_train, y_train)
                return self._create_success_response("Ensemble trained", result)
            
            elif operation == "predict_ensemble":
                ensemble = data.get("ensemble")
                X = np.array(data.get("X", []))
                
                predictions = self.ensemble_methods.predict(ensemble, X)
                return self._create_success_response("Ensemble predictions made", {"predictions": predictions.tolist()})
            
            else:
                return self._create_error_response(f"Unknown ensemble operation: {operation}")
                
        except Exception as e:
            return self._create_error_response(f"Error in ensemble operation: {str(e)}")
    
    def _handle_time_series_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle time series requests."""
        try:
            operation = request.get("operation", "")
            data = request.get("data", {})
            
            if operation == "create_time_series_model":
                model_type = data.get("model_type")
                config = data.get("config", {})
                
                model = self.time_series_models.create_model(model_type, config)
                return self._create_success_response("Time series model created", {"model": str(model)})
            
            elif operation == "forecast":
                model = data.get("model")
                data_series = np.array(data.get("data_series", []))
                horizon = data.get("horizon", 30)
                
                forecast = self.time_series_models.forecast(model, data_series, horizon)
                return self._create_success_response("Forecast generated", {"forecast": forecast.tolist()})
            
            elif operation == "analyze_seasonality":
                data_series = np.array(data.get("data_series", []))
                
                analysis = self.time_series_models.analyze_seasonality(data_series)
                return self._create_success_response("Seasonality analyzed", analysis)
            
            else:
                return self._create_error_response(f"Unknown time series operation: {operation}")
                
        except Exception as e:
            return self._create_error_response(f"Error in time series operation: {str(e)}")
    
    def _handle_clustering_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clustering requests."""
        try:
            operation = request.get("operation", "")
            data = request.get("data", {})
            
            if operation == "create_clustering_model":
                algorithm = data.get("algorithm")
                config = data.get("config", {})
                
                model = self.clustering_algorithms.create_model(algorithm, config)
                return self._create_success_response("Clustering model created", {"model": str(model)})
            
            elif operation == "cluster_data":
                model = data.get("model")
                X = np.array(data.get("X", []))
                
                clusters = self.clustering_algorithms.cluster_data(model, X)
                return self._create_success_response("Data clustered", {"clusters": clusters.tolist()})
            
            elif operation == "analyze_clusters":
                X = np.array(data.get("X", []))
                clusters = np.array(data.get("clusters", []))
                
                analysis = self.clustering_algorithms.analyze_clusters(X, clusters)
                return self._create_success_response("Clusters analyzed", analysis)
            
            else:
                return self._create_error_response(f"Unknown clustering operation: {operation}")
                
        except Exception as e:
            return self._create_error_response(f"Error in clustering operation: {str(e)}")
    
    def _handle_dimensionality_reduction_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dimensionality reduction requests."""
        try:
            operation = request.get("operation", "")
            data = request.get("data", {})
            
            if operation == "reduce_dimensions":
                method = data.get("method")
                X = np.array(data.get("X", []))
                config = data.get("config", {})
                
                reduced_data = self.dimensionality_reduction.reduce_dimensions(method, X, config)
                return self._create_success_response("Dimensions reduced", {"reduced_data": reduced_data.tolist()})
            
            elif operation == "visualize_reduced_data":
                reduced_data = np.array(data.get("reduced_data", []))
                labels = data.get("labels", [])
                
                visualization = self.dimensionality_reduction.visualize_data(reduced_data, labels)
                return self._create_success_response("Visualization created", visualization)
            
            else:
                return self._create_error_response(f"Unknown dimensionality reduction operation: {operation}")
                
        except Exception as e:
            return self._create_error_response(f"Error in dimensionality reduction operation: {str(e)}")
    
    def _handle_automl_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AutoML requests."""
        try:
            operation = request.get("operation", "")
            data = request.get("data", {})
            
            if operation == "run_automl":
                X_train = np.array(data.get("X_train", []))
                y_train = np.array(data.get("y_train", []))
                X_test = np.array(data.get("X_test", [])) if data.get("X_test") else None
                y_test = np.array(data.get("y_test", [])) if data.get("y_test") else None
                config = data.get("config", {})
                
                result = self.automl_pipeline.run_automl(X_train, y_train, X_test, y_test, config)
                return self._create_success_response("AutoML completed", result)
            
            elif operation == "get_best_model":
                automl_result = data.get("automl_result")
                
                best_model = self.automl_pipeline.get_best_model(automl_result)
                return self._create_success_response("Best model retrieved", {"best_model": str(best_model)})
            
            elif operation == "optimize_hyperparameters":
                model = data.get("model")
                X_train = np.array(data.get("X_train", []))
                y_train = np.array(data.get("y_train", []))
                param_space = data.get("param_space", {})
                
                optimized_model = self.automl_pipeline.optimize_hyperparameters(
                    model, X_train, y_train, param_space
                )
                return self._create_success_response("Hyperparameters optimized", {"optimized_model": str(optimized_model)})
            
            else:
                return self._create_error_response(f"Unknown AutoML operation: {operation}")
                
        except Exception as e:
            return self._create_error_response(f"Error in AutoML operation: {str(e)}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities."""
        return {
            "agent_name": self.agent_name,
            "agent_description": self.agent_description,
            "capabilities": self.capabilities,
            "config": self.config.get_config()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_name": self.agent_name,
            "status": "active",
            "components_loaded": {
                "deep_learning_engine": self.deep_learning_engine is not None,
                "transfer_learning_service": self.transfer_learning_service is not None,
                "model_versioning": self.model_versioning is not None,
                "ensemble_methods": self.ensemble_methods is not None,
                "time_series_models": self.time_series_models is not None,
                "clustering_algorithms": self.clustering_algorithms is not None,
                "dimensionality_reduction": self.dimensionality_reduction is not None,
                "automl_pipeline": self.automl_pipeline is not None
            },
            "last_updated": datetime.now().isoformat()
        }
