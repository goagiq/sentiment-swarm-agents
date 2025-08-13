"""
Causal Inference Engine

Advanced causal inference capabilities for identifying cause-effect relationships
and performing causal analysis in multivariate datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

# Conditional imports for causal inference libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Causal inference features limited.")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.stats.diagnostic import het_breuschpagan
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Some causal tests not supported.")

# Local imports
from ..error_handling_service import ErrorHandlingService
from ..caching_service import CachingService
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class CausalRelationship:
    """Data class for causal relationships"""
    cause: str
    effect: str
    strength: float
    confidence: float
    method: str
    p_value: Optional[float] = None
    direction: str = "positive"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CausalAnalysisResult:
    """Result container for causal analysis"""
    relationships: List[CausalRelationship]
    causal_graph: Dict[str, List[str]]
    strength_matrix: pd.DataFrame
    confidence_matrix: pd.DataFrame
    analysis_methods: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CausalInferenceEngine:
    """
    Advanced causal inference engine for identifying cause-effect relationships
    and performing sophisticated causal analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the causal inference engine"""
        self.config = config or {}
        self.error_handler = ErrorHandlingService()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Analysis storage
        self.causal_relationships = []
        self.analysis_history = []
        
        # Configuration
        self.significance_level = self.config.get('significance_level', 0.05)
        self.min_relationship_strength = self.config.get('min_relationship_strength', 0.1)
        self.max_lag = self.config.get('max_lag', 10)
        
        logger.info("CausalInferenceEngine initialized")
    
    def detect_granger_causality(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        max_lag: Optional[int] = None
    ) -> List[CausalRelationship]:
        """
        Detect Granger causality relationships between variables
        
        Args:
            data: Time series data
            variables: Variables to analyze (if None, use all numeric columns)
            max_lag: Maximum lag for causality tests
            
        Returns:
            List of causal relationships
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available for Granger causality")
            return []
        
        try:
            with self.performance_monitor.track_operation("granger_causality_detection"):
                if variables is None:
                    variables = data.select_dtypes(include=[np.number]).columns.tolist()
                
                if max_lag is None:
                    max_lag = self.max_lag
                
                relationships = []
                
                for cause in variables:
                    for effect in variables:
                        if cause != effect:
                            try:
                                # Prepare data for Granger causality test
                                test_data = data[[effect, cause]].dropna()
                                if len(test_data) < max_lag + 10:  # Need sufficient data
                                    continue
                                
                                # Perform Granger causality test
                                gc_result = grangercausalitytests(
                                    test_data, maxlag=max_lag, verbose=False
                                )
                                
                                # Extract p-values
                                p_values = []
                                for lag in range(1, max_lag + 1):
                                    if lag in gc_result:
                                        p_value = gc_result[lag][0]['ssr_chi2test'][1]
                                        p_values.append(p_value)
                                
                                if p_values:
                                    min_p_value = min(p_values)
                                    if min_p_value < self.significance_level:
                                        # Calculate relationship strength
                                        strength = 1.0 - min_p_value
                                        confidence = 1.0 - min_p_value
                                        
                                        relationship = CausalRelationship(
                                            cause=cause,
                                            effect=effect,
                                            strength=strength,
                                            confidence=confidence,
                                            method="granger_causality",
                                            p_value=min_p_value
                                        )
                                        relationships.append(relationship)
                                        
                            except Exception as e:
                                logger.debug(f"Error in Granger causality test for {cause} -> {effect}: {str(e)}")
                                continue
                
                logger.info(f"Detected {len(relationships)} Granger causality relationships")
                return relationships
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in Granger causality detection: {str(e)}", e)
            return []
    
    def detect_correlation_causality(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        method: str = "pearson"
    ) -> List[CausalRelationship]:
        """
        Detect causal relationships using correlation analysis
        
        Args:
            data: Input data
            variables: Variables to analyze
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            List of causal relationships
        """
        try:
            with self.performance_monitor.track_operation("correlation_causality_detection"):
                if variables is None:
                    variables = data.select_dtypes(include=[np.number]).columns.tolist()
                
                # Calculate correlation matrix
                corr_matrix = data[variables].corr(method=method)
                
                relationships = []
                
                for cause in variables:
                    for effect in variables:
                        if cause != effect:
                            correlation = corr_matrix.loc[cause, effect]
                            
                            if abs(correlation) >= self.min_relationship_strength:
                                strength = abs(correlation)
                                confidence = abs(correlation)  # Simplified confidence
                                direction = "positive" if correlation > 0 else "negative"
                                
                                relationship = CausalRelationship(
                                    cause=cause,
                                    effect=effect,
                                    strength=strength,
                                    confidence=confidence,
                                    method=f"{method}_correlation",
                                    direction=direction
                                )
                                relationships.append(relationship)
                
                logger.info(f"Detected {len(relationships)} correlation-based relationships")
                return relationships
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in correlation causality detection: {str(e)}", e)
            return []
    
    def detect_conditional_independence(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        conditioning_vars: Optional[List[str]] = None
    ) -> List[CausalRelationship]:
        """
        Detect conditional independence relationships
        
        Args:
            data: Input data
            variables: Variables to analyze
            conditioning_vars: Conditioning variables
            
        Returns:
            List of causal relationships
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for conditional independence")
            return []
        
        try:
            with self.performance_monitor.track_operation("conditional_independence_detection"):
                if variables is None:
                    variables = data.select_dtypes(include=[np.number]).columns.tolist()
                
                if conditioning_vars is None:
                    conditioning_vars = []
                
                relationships = []
                
                for var1 in variables:
                    for var2 in variables:
                        if var1 != var2 and var1 not in conditioning_vars and var2 not in conditioning_vars:
                            try:
                                # Calculate unconditional correlation
                                unconditional_corr = data[var1].corr(data[var2])
                                
                                # Calculate conditional correlation
                                if conditioning_vars:
                                    # Use linear regression to condition on other variables
                                    X = data[conditioning_vars]
                                    y1 = data[var1]
                                    y2 = data[var2]
                                    
                                    # Fit models
                                    model1 = LinearRegression()
                                    model2 = LinearRegression()
                                    
                                    model1.fit(X, y1)
                                    model2.fit(X, y2)
                                    
                                    # Get residuals
                                    residuals1 = y1 - model1.predict(X)
                                    residuals2 = y2 - model2.predict(X)
                                    
                                    conditional_corr = residuals1.corr(residuals2)
                                else:
                                    conditional_corr = unconditional_corr
                                
                                # Calculate partial correlation
                                if abs(unconditional_corr) > abs(conditional_corr):
                                    strength = abs(unconditional_corr - conditional_corr)
                                    confidence = abs(unconditional_corr)
                                    
                                    if strength >= self.min_relationship_strength:
                                        relationship = CausalRelationship(
                                            cause=var1,
                                            effect=var2,
                                            strength=strength,
                                            confidence=confidence,
                                            method="conditional_independence",
                                            direction="positive" if unconditional_corr > 0 else "negative"
                                        )
                                        relationships.append(relationship)
                                        
                            except Exception as e:
                                logger.debug(f"Error in conditional independence test for {var1} -> {var2}: {str(e)}")
                                continue
                
                logger.info(f"Detected {len(relationships)} conditional independence relationships")
                return relationships
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in conditional independence detection: {str(e)}", e)
            return []
    
    def perform_causal_analysis(
        self,
        data: pd.DataFrame,
        methods: Optional[List[str]] = None,
        variables: Optional[List[str]] = None
    ) -> CausalAnalysisResult:
        """
        Perform comprehensive causal analysis using multiple methods
        
        Args:
            data: Input data
            methods: Analysis methods to use
            variables: Variables to analyze
            
        Returns:
            CausalAnalysisResult object
        """
        try:
            with self.performance_monitor.track_operation("comprehensive_causal_analysis"):
                if methods is None:
                    methods = ["correlation", "granger_causality", "conditional_independence"]
                
                all_relationships = []
                
                # Perform different types of causal analysis
                if "correlation" in methods:
                    corr_relationships = self.detect_correlation_causality(data, variables)
                    all_relationships.extend(corr_relationships)
                
                if "granger_causality" in methods and data.index.dtype == 'datetime64[ns]':
                    granger_relationships = self.detect_granger_causality(data, variables)
                    all_relationships.extend(granger_relationships)
                
                if "conditional_independence" in methods:
                    cond_relationships = self.detect_conditional_independence(data, variables)
                    all_relationships.extend(cond_relationships)
                
                # Create causal graph
                causal_graph = self._create_causal_graph(all_relationships)
                
                # Create strength and confidence matrices
                strength_matrix, confidence_matrix = self._create_matrices(
                    all_relationships, variables or data.columns.tolist()
                )
                
                # Create result
                result = CausalAnalysisResult(
                    relationships=all_relationships,
                    causal_graph=causal_graph,
                    strength_matrix=strength_matrix,
                    confidence_matrix=confidence_matrix,
                    analysis_methods=methods
                )
                
                # Store in history
                self.analysis_history.append(result)
                self.causal_relationships.extend(all_relationships)
                
                logger.info(f"Completed causal analysis with {len(all_relationships)} relationships")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in causal analysis: {str(e)}", e)
            raise
    
    def _create_causal_graph(self, relationships: List[CausalRelationship]) -> Dict[str, List[str]]:
        """Create causal graph from relationships"""
        causal_graph = {}
        
        for rel in relationships:
            if rel.cause not in causal_graph:
                causal_graph[rel.cause] = []
            causal_graph[rel.cause].append(rel.effect)
        
        return causal_graph
    
    def _create_matrices(
        self,
        relationships: List[CausalRelationship],
        variables: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create strength and confidence matrices"""
        strength_matrix = pd.DataFrame(0.0, index=variables, columns=variables)
        confidence_matrix = pd.DataFrame(0.0, index=variables, columns=variables)
        
        for rel in relationships:
            if rel.cause in variables and rel.effect in variables:
                strength_matrix.loc[rel.cause, rel.effect] = rel.strength
                confidence_matrix.loc[rel.cause, rel.effect] = rel.confidence
        
        return strength_matrix, confidence_matrix
    
    def get_strongest_relationships(
        self,
        min_strength: Optional[float] = None,
        min_confidence: Optional[float] = None,
        method: Optional[str] = None
    ) -> List[CausalRelationship]:
        """Get strongest causal relationships"""
        if min_strength is None:
            min_strength = self.min_relationship_strength
        
        filtered_relationships = []
        
        for rel in self.causal_relationships:
            if (rel.strength >= min_strength and
                (min_confidence is None or rel.confidence >= min_confidence) and
                (method is None or rel.method == method)):
                filtered_relationships.append(rel)
        
        # Sort by strength
        filtered_relationships.sort(key=lambda x: x.strength, reverse=True)
        
        return filtered_relationships
    
    def get_causal_paths(
        self,
        start_variable: str,
        end_variable: str,
        max_path_length: int = 3
    ) -> List[List[str]]:
        """Find causal paths between two variables"""
        causal_graph = self._create_causal_graph(self.causal_relationships)
        
        def find_paths(current: str, target: str, path: List[str], max_length: int) -> List[List[str]]:
            if len(path) > max_length:
                return []
            
            if current == target:
                return [path]
            
            paths = []
            for neighbor in causal_graph.get(current, []):
                if neighbor not in path:
                    new_paths = find_paths(neighbor, target, path + [neighbor], max_length)
                    paths.extend(new_paths)
            
            return paths
        
        return find_paths(start_variable, end_variable, [start_variable], max_path_length)
    
    def export_analysis(self, filepath: str) -> None:
        """Export causal analysis results"""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                'relationships': [
                    {
                        'cause': rel.cause,
                        'effect': rel.effect,
                        'strength': rel.strength,
                        'confidence': rel.confidence,
                        'method': rel.method,
                        'p_value': rel.p_value,
                        'direction': rel.direction,
                        'timestamp': rel.timestamp.isoformat() if rel.timestamp else None
                    }
                    for rel in self.causal_relationships
                ],
                'analysis_history': [
                    {
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                        'methods': result.analysis_methods,
                        'relationship_count': len(result.relationships)
                    }
                    for result in self.analysis_history
                ],
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Causal analysis exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting causal analysis: {str(e)}", e)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of causal analysis results"""
        if not self.causal_relationships:
            return {"message": "No causal relationships found"}
        
        summary = {
            'total_relationships': len(self.causal_relationships),
            'methods_used': list(set(rel.method for rel in self.causal_relationships)),
            'strongest_relationship': max(self.causal_relationships, key=lambda x: x.strength),
            'average_strength': np.mean([rel.strength for rel in self.causal_relationships]),
            'average_confidence': np.mean([rel.confidence for rel in self.causal_relationships]),
            'analysis_count': len(self.analysis_history)
        }
        
        return summary
