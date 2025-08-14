"""
Advanced Anomaly Detection Agent

Specialized agent for detecting anomalies and unusual patterns in multivariate data
using advanced algorithms and statistical methods.
"""

from typing import Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd

from loguru import logger

from .base_agent import StrandsBaseAgent
from ..core.models import AnalysisRequest, AnalysisResult, DataType
from ..core.advanced_analytics.advanced_anomaly_detection import (
    AdvancedAnomalyDetector
)
from ..core.error_handling_service import ErrorHandlingService
from ..core.caching_service import CachingService
from ..core.performance_monitor import PerformanceMonitor
from ..config.advanced_analytics_config import AdvancedAnalyticsConfig


class AnomalyDetectionAgent(StrandsBaseAgent):
    """
    Advanced anomaly detection agent for identifying outliers and unusual patterns
    in multivariate datasets using sophisticated algorithms.
    """

    def __init__(self, model_name: str = None):
        """Initialize the anomaly detection agent"""
        super().__init__(model_name=model_name)
        
        # Initialize services
        self.error_handler = ErrorHandlingService()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize advanced analytics components
        self.config = AdvancedAnalyticsConfig()
        # Convert config to dict for compatibility
        anomaly_config_dict = {
            'contamination': self.config.anomaly_detection.contamination,
            'threshold_method': self.config.anomaly_detection.threshold_method,
            'severity_thresholds': self.config.anomaly_detection.severity_thresholds,
            'isolation_forest_n_estimators': self.config.anomaly_detection.isolation_forest_n_estimators,
            'one_class_svm_kernel': self.config.anomaly_detection.one_class_svm_kernel,
            'one_class_svm_nu': self.config.anomaly_detection.one_class_svm_nu,
            'lof_n_neighbors': self.config.anomaly_detection.lof_n_neighbors,
            'statistical_threshold': self.config.anomaly_detection.statistical_threshold
        }
        self.anomaly_detector = AdvancedAnomalyDetector(anomaly_config_dict)
        
        # Set agent metadata
        self.metadata.update({
            "agent_type": "anomaly_detection",
            "capabilities": [
                "multivariate_anomaly_detection",
                "statistical_outlier_detection",
                "isolation_forest_detection",
                "one_class_svm_detection",
                "local_outlier_factor_detection",
                "real_time_monitoring",
                "anomaly_scoring",
                "severity_assessment"
            ],
            "supported_methods": [
                "isolation_forest",
                "one_class_svm", 
                "local_outlier_factor",
                "statistical"
            ],
            "data_types": [DataType.TEXT, DataType.NUMERICAL],
            "version": "1.0.0"
        })
        
        logger.info(f"AnomalyDetectionAgent {self.agent_id} initialized")

    def _get_tools(self) -> list:
        """Get list of tools for this agent"""
        return [
            self.detect_anomalies,
            self.detect_multivariate_anomalies,
            self.detect_statistical_outliers,
            self.detect_isolation_forest,
            self.detect_one_class_svm,
            self.detect_local_outlier_factor,
            self.get_anomaly_summary,
            self.export_anomaly_results,
            self.monitor_real_time_anomalies
        ]

    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request"""
        return (
            request.data_type in [DataType.TEXT, DataType.NUMERICAL] and
            "anomaly" in request.analysis_type.lower()
        )

    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request"""
        try:
            with self.performance_monitor.track_operation("anomaly_detection_processing"):
                # Extract data
                data = self._extract_data(request)
                
                # Determine detection method
                method = self._determine_method(request)
                
                # Perform anomaly detection
                if method == "multivariate":
                    result = await self.detect_multivariate_anomalies(data, request.parameters)
                elif method == "statistical":
                    result = await self.detect_statistical_outliers(data, request.parameters)
                elif method == "isolation_forest":
                    result = await self.detect_isolation_forest(data, request.parameters)
                elif method == "one_class_svm":
                    result = await self.detect_one_class_svm(data, request.parameters)
                elif method == "local_outlier_factor":
                    result = await self.detect_local_outlier_factor(data, request.parameters)
                else:
                    result = await self.detect_anomalies(data, request.parameters)
                
                # Create analysis result
                analysis_result = AnalysisResult(
                    request_id=request.id,
                    agent_id=self.agent_id,
                    result_type="anomaly_detection",
                    data=result,
                    metadata={
                        "method_used": method,
                        "anomaly_count": len(result.get("anomalies", [])),
                        "processing_time": self.performance_monitor.get_last_operation_time(),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                logger.info(f"Anomaly detection completed: {len(result.get('anomalies', []))} anomalies found")
                return analysis_result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in anomaly detection processing: {str(e)}", e)
            raise

    async def detect_anomalies(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using the best available method
        
        Args:
            data: Input data (DataFrame or dict)
            parameters: Detection parameters
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            with self.performance_monitor.track_operation("anomaly_detection"):
                # Convert data to DataFrame if needed
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                
                # Use default parameters if none provided
                if parameters is None:
                    parameters = {}
                
                # Perform multivariate anomaly detection
                results = self.anomaly_detector.detect_multivariate_anomalies(
                    df,
                    methods=parameters.get("methods", ["isolation_forest", "statistical"]),
                    ensemble=parameters.get("ensemble", True)
                )
                
                # Format results
                formatted_results = {
                    "anomalies": [],
                    "anomaly_scores": {},
                    "methods_used": list(results.keys()),
                    "summary": {}
                }
                
                # Combine results from all methods
                for method, result in results.items():
                    formatted_results["anomalies"].extend([
                        {
                            "index": anomaly.index,
                            "value": anomaly.value,
                            "score": anomaly.score,
                            "method": anomaly.method,
                            "severity": anomaly.severity,
                            "timestamp": anomaly.timestamp.isoformat() if anomaly.timestamp else None
                        }
                        for anomaly in result.anomalies
                    ])
                    
                    # Add anomaly scores
                    if hasattr(result, 'anomaly_scores'):
                        formatted_results["anomaly_scores"][method] = result.anomaly_scores.to_dict()
                
                # Get summary
                formatted_results["summary"] = self.anomaly_detector.get_detection_summary()
                
                return formatted_results
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in anomaly detection: {str(e)}", e)
            raise

    async def detect_multivariate_anomalies(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies in multivariate data using multiple methods
        
        Args:
            data: Input data
            parameters: Detection parameters
            
        Returns:
            Dictionary containing multivariate anomaly detection results
        """
        try:
            with self.performance_monitor.track_operation("multivariate_anomaly_detection"):
                # Convert data to DataFrame if needed
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                
                # Use default parameters if none provided
                if parameters is None:
                    parameters = {}
                
                # Perform multivariate detection
                results = self.anomaly_detector.detect_multivariate_anomalies(
                    df,
                    methods=parameters.get("methods", ["isolation_forest", "one_class_svm", "statistical"]),
                    ensemble=parameters.get("ensemble", True)
                )
                
                # Format results
                formatted_results = {
                    "multivariate_results": {},
                    "ensemble_result": None,
                    "summary": {}
                }
                
                for method, result in results.items():
                    formatted_results["multivariate_results"][method] = {
                        "anomaly_count": len(result.anomalies),
                        "threshold": result.threshold,
                        "performance_metrics": result.performance_metrics,
                        "method_used": result.method_used
                    }
                    
                    if method == "ensemble":
                        formatted_results["ensemble_result"] = {
                            "anomaly_count": len(result.anomalies),
                            "threshold": result.threshold,
                            "performance_metrics": result.performance_metrics
                        }
                
                # Get summary
                formatted_results["summary"] = self.anomaly_detector.get_detection_summary()
                
                return formatted_results
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in multivariate anomaly detection: {str(e)}", e)
            raise

    async def detect_statistical_outliers(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods
        
        Args:
            data: Input data
            parameters: Detection parameters
            
        Returns:
            Dictionary containing statistical anomaly detection results
        """
        try:
            with self.performance_monitor.track_operation("statistical_anomaly_detection"):
                # Convert data to DataFrame if needed
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                
                # Use default parameters if none provided
                if parameters is None:
                    parameters = {}
                
                # Perform statistical detection
                result = self.anomaly_detector.detect_statistical_outliers(
                    df,
                    method=parameters.get("method", "zscore"),
                    threshold=parameters.get("threshold", 3.0)
                )
                
                # Format results
                formatted_results = {
                    "anomalies": [
                        {
                            "index": anomaly.index,
                            "value": anomaly.value,
                            "score": anomaly.score,
                            "method": anomaly.method,
                            "severity": anomaly.severity,
                            "timestamp": anomaly.timestamp.isoformat() if anomaly.timestamp else None
                        }
                        for anomaly in result.anomalies
                    ],
                    "anomaly_scores": result.anomaly_scores.to_dict(),
                    "threshold": result.threshold,
                    "method_used": result.method_used,
                    "performance_metrics": result.performance_metrics
                }
                
                return formatted_results
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in statistical anomaly detection: {str(e)}", e)
            raise

    async def detect_isolation_forest(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            data: Input data
            parameters: Detection parameters
            
        Returns:
            Dictionary containing Isolation Forest results
        """
        try:
            with self.performance_monitor.track_operation("isolation_forest_detection"):
                # Convert data to DataFrame if needed
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                
                # Use default parameters if none provided
                if parameters is None:
                    parameters = {}
                
                # Perform Isolation Forest detection
                result = self.anomaly_detector.detect_isolation_forest(
                    df,
                    contamination=parameters.get("contamination", 0.1)
                )
                
                # Format results
                formatted_results = {
                    "anomalies": [
                        {
                            "index": anomaly.index,
                            "value": anomaly.value,
                            "score": anomaly.score,
                            "method": anomaly.method,
                            "severity": anomaly.severity,
                            "timestamp": anomaly.timestamp.isoformat() if anomaly.timestamp else None
                        }
                        for anomaly in result.anomalies
                    ],
                    "anomaly_scores": result.anomaly_scores.to_dict(),
                    "threshold": result.threshold,
                    "method_used": result.method_used,
                    "performance_metrics": result.performance_metrics
                }
                
                return formatted_results
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in Isolation Forest detection: {str(e)}", e)
            raise

    async def detect_one_class_svm(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using One-Class SVM
        
        Args:
            data: Input data
            parameters: Detection parameters
            
        Returns:
            Dictionary containing One-Class SVM results
        """
        try:
            with self.performance_monitor.track_operation("one_class_svm_detection"):
                # Convert data to DataFrame if needed
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                
                # Use default parameters if none provided
                if parameters is None:
                    parameters = {}
                
                # Perform One-Class SVM detection
                result = self.anomaly_detector.detect_one_class_svm(
                    df,
                    kernel=parameters.get("kernel", "rbf"),
                    nu=parameters.get("nu", 0.1)
                )
                
                # Format results
                formatted_results = {
                    "anomalies": [
                        {
                            "index": anomaly.index,
                            "value": anomaly.value,
                            "score": anomaly.score,
                            "method": anomaly.method,
                            "severity": anomaly.severity,
                            "timestamp": anomaly.timestamp.isoformat() if anomaly.timestamp else None
                        }
                        for anomaly in result.anomalies
                    ],
                    "anomaly_scores": result.anomaly_scores.to_dict(),
                    "threshold": result.threshold,
                    "method_used": result.method_used,
                    "performance_metrics": result.performance_metrics
                }
                
                return formatted_results
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in One-Class SVM detection: {str(e)}", e)
            raise

    async def detect_local_outlier_factor(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using Local Outlier Factor
        
        Args:
            data: Input data
            parameters: Detection parameters
            
        Returns:
            Dictionary containing Local Outlier Factor results
        """
        try:
            with self.performance_monitor.track_operation("lof_detection"):
                # Convert data to DataFrame if needed
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                
                # Use default parameters if none provided
                if parameters is None:
                    parameters = {}
                
                # Perform Local Outlier Factor detection
                result = self.anomaly_detector.detect_local_outlier_factor(
                    df,
                    n_neighbors=parameters.get("n_neighbors", 20),
                    contamination=parameters.get("contamination", 0.1)
                )
                
                # Format results
                formatted_results = {
                    "anomalies": [
                        {
                            "index": anomaly.index,
                            "value": anomaly.value,
                            "score": anomaly.score,
                            "method": anomaly.method,
                            "severity": anomaly.severity,
                            "timestamp": anomaly.timestamp.isoformat() if anomaly.timestamp else None
                        }
                        for anomaly in result.anomalies
                    ],
                    "anomaly_scores": result.anomaly_scores.to_dict(),
                    "threshold": result.threshold,
                    "method_used": result.method_used,
                    "performance_metrics": result.performance_metrics
                }
                
                return formatted_results
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in Local Outlier Factor detection: {str(e)}", e)
            raise

    async def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get summary of anomaly detection results
        
        Returns:
            Dictionary containing anomaly detection summary
        """
        try:
            summary = self.anomaly_detector.get_detection_summary()
            return {
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id
            }
        except Exception as e:
            self.error_handler.handle_error(f"Error getting anomaly summary: {str(e)}", e)
            raise

    async def export_anomaly_results(
        self,
        filepath: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export anomaly detection results
        
        Args:
            filepath: Path to export file
            format: Export format
            
        Returns:
            Dictionary containing export status
        """
        try:
            with self.performance_monitor.track_operation("anomaly_results_export"):
                self.anomaly_detector.export_detection_results(filepath)
                
                return {
                    "status": "success",
                    "filepath": filepath,
                    "format": format,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting anomaly results: {str(e)}", e)
            raise

    async def monitor_real_time_anomalies(
        self,
        data_stream: Any,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Monitor real-time data stream for anomalies
        
        Args:
            data_stream: Real-time data stream
            parameters: Monitoring parameters
            
        Returns:
            Dictionary containing real-time monitoring results
        """
        try:
            with self.performance_monitor.track_operation("real_time_anomaly_monitoring"):
                # This is a placeholder for real-time monitoring
                # In a real implementation, this would connect to a data stream
                # and continuously monitor for anomalies
                
                return {
                    "status": "monitoring_active",
                    "message": "Real-time anomaly monitoring not yet implemented",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.error_handler.handle_error(f"Error in real-time anomaly monitoring: {str(e)}", e)
            raise

    def _extract_data(self, request: AnalysisRequest) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Extract data from the analysis request"""
        if request.data_type == DataType.TABULAR:
            return pd.DataFrame(request.data)
        elif request.data_type == DataType.NUMERICAL:
            return pd.DataFrame(request.data)
        else:
            return request.data

    def _determine_method(self, request: AnalysisRequest) -> str:
        """Determine the anomaly detection method to use"""
        analysis_type = request.analysis_type.lower()
        
        if "multivariate" in analysis_type:
            return "multivariate"
        elif "statistical" in analysis_type:
            return "statistical"
        elif "isolation" in analysis_type or "forest" in analysis_type:
            return "isolation_forest"
        elif "svm" in analysis_type:
            return "one_class_svm"
        elif "lof" in analysis_type or "local" in analysis_type:
            return "local_outlier_factor"
        else:
            return "auto"  # Use best available method
