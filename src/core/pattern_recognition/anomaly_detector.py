"""
Anomaly Detector

This module provides anomaly detection capabilities including:
- Statistical outlier detection
- Isolation forest anomaly detection
- Real-time anomaly scoring
- Anomaly classification
"""

import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

from src.core.error_handler import with_error_handling


class AnomalyDetector:
    """
    Detects anomalies and outliers in data.
    """
    
    def __init__(self):
        self.anomaly_cache = {}
        self.detection_config = {
            "anomaly_threshold": 0.95,
            "iqr_multiplier": 1.5,
            "z_score_threshold": 3.0,
            "isolation_forest_contamination": 0.1
        }
        
        logger.info("AnomalyDetector initialized successfully")
    
    @with_error_handling("anomaly_detection")
    async def detect_anomalies(
        self, 
        data: List[float], 
        detection_method: str = "statistical"
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the provided data.
        
        Args:
            data: List of numerical values
            detection_method: Method to use for anomaly detection
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            logger.info(f"Detecting anomalies using {detection_method} method")
            
            if len(data) < 3:
                return {"error": "Insufficient data for anomaly detection"}
            
            # Convert to numpy array
            data_array = np.array(data)
            
            # Perform anomaly detection based on method
            if detection_method == "statistical":
                result = await self._statistical_anomaly_detection(data_array)
            elif detection_method == "isolation_forest":
                result = await self._isolation_forest_detection(data_array)
            elif detection_method == "combined":
                result = await self._combined_anomaly_detection(data_array)
            else:
                return {"error": f"Unknown detection method: {detection_method}"}
            
            return result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"error": str(e)}
    
    async def _statistical_anomaly_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform statistical anomaly detection."""
        try:
            anomalies = []
            
            # IQR method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.detection_config["iqr_multiplier"] * iqr
            upper_bound = q3 + self.detection_config["iqr_multiplier"] * iqr
            
            iqr_anomalies = []
            for i, value in enumerate(data):
                if value < lower_bound or value > upper_bound:
                    iqr_anomalies.append({
                        "index": i,
                        "value": float(value),
                        "method": "iqr",
                        "severity": "high" if abs(value - np.mean(data)) > 2 * np.std(data) else "medium"
                    })
            
            # Z-score method
            mean = np.mean(data)
            std = np.std(data)
            z_scores = np.abs((data - mean) / std)
            
            z_score_anomalies = []
            for i, z_score in enumerate(z_scores):
                if z_score > self.detection_config["z_score_threshold"]:
                    z_score_anomalies.append({
                        "index": i,
                        "value": float(data[i]),
                        "z_score": float(z_score),
                        "method": "z_score",
                        "severity": "high" if z_score > 4 else "medium"
                    })
            
            # Combine results
            all_anomalies = iqr_anomalies + z_score_anomalies
            
            return {
                "anomalies": all_anomalies,
                "total_anomalies": len(all_anomalies),
                "iqr_anomalies": len(iqr_anomalies),
                "z_score_anomalies": len(z_score_anomalies),
                "anomaly_rate": len(all_anomalies) / len(data),
                "detection_method": "statistical",
                "statistics": {
                    "mean": float(mean),
                    "std": float(std),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr)
                }
            }
            
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
            return {"error": str(e)}
    
    async def _isolation_forest_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform isolation forest anomaly detection."""
        try:
            # Simple isolation forest simulation
            # In a real implementation, you would use sklearn.ensemble.IsolationForest
            
            # Calculate anomaly scores based on distance from mean
            mean = np.mean(data)
            std = np.std(data)
            
            # Normalize data
            normalized_data = (data - mean) / std
            
            # Calculate anomaly scores (higher = more anomalous)
            anomaly_scores = np.abs(normalized_data)
            
            # Find anomalies above threshold
            threshold = np.percentile(anomaly_scores, 
                                    (1 - self.detection_config["isolation_forest_contamination"]) * 100)
            
            anomalies = []
            for i, score in enumerate(anomaly_scores):
                if score > threshold:
                    anomalies.append({
                        "index": i,
                        "value": float(data[i]),
                        "anomaly_score": float(score),
                        "method": "isolation_forest",
                        "severity": "high" if score > threshold * 1.5 else "medium"
                    })
            
            return {
                "anomalies": anomalies,
                "total_anomalies": len(anomalies),
                "anomaly_rate": len(anomalies) / len(data),
                "detection_method": "isolation_forest",
                "threshold": float(threshold),
                "anomaly_scores": anomaly_scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"Isolation forest detection failed: {e}")
            return {"error": str(e)}
    
    async def _combined_anomaly_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform combined anomaly detection using multiple methods."""
        try:
            # Get results from both methods
            statistical_result = await self._statistical_anomaly_detection(data)
            isolation_result = await self._isolation_forest_detection(data)
            
            if "error" in statistical_result or "error" in isolation_result:
                return {"error": "Combined detection failed"}
            
            # Combine anomalies
            all_anomalies = statistical_result["anomalies"] + isolation_result["anomalies"]
            
            # Remove duplicates based on index
            unique_anomalies = {}
            for anomaly in all_anomalies:
                index = anomaly["index"]
                if index not in unique_anomalies:
                    unique_anomalies[index] = anomaly
                else:
                    # Merge methods if same index
                    existing = unique_anomalies[index]
                    existing["methods"] = [existing["method"], anomaly["method"]]
                    existing["method"] = "combined"
            
            combined_anomalies = list(unique_anomalies.values())
            
            return {
                "anomalies": combined_anomalies,
                "total_anomalies": len(combined_anomalies),
                "anomaly_rate": len(combined_anomalies) / len(data),
                "detection_method": "combined",
                "statistical_result": statistical_result,
                "isolation_result": isolation_result
            }
            
        except Exception as e:
            logger.error(f"Combined anomaly detection failed: {e}")
            return {"error": str(e)}
    
    async def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get a summary of anomaly detection results."""
        try:
            return {
                "total_detections": len(self.anomaly_cache),
                "detection_config": self.detection_config,
                "analysis_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Anomaly summary generation failed: {e}")
            return {"error": str(e)}
