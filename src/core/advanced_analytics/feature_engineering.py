"""
Feature Engineering

Advanced feature engineering capabilities for automated feature creation,
transformation, and selection in machine learning pipelines.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

# Conditional imports
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Feature engineering limited.")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Statistical features limited.")

# Local imports
from ..error_handler import ErrorHandler
from ..caching_service import CachingService
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Data class for feature engineering results"""
    engineered_features: pd.DataFrame
    feature_names: List[str]
    feature_types: Dict[str, str]
    transformation_info: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FeatureEngineer:
    """
    Advanced feature engineering for automated feature creation and transformation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature engineer"""
        self.config = config or {}
        self.error_handler = ErrorHandler()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Configuration
        self.max_features = self.config.get('max_features', 100)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.95)
        self.feature_importance_threshold = self.config.get('feature_importance_threshold', 0.01)
        
        # Storage
        self.engineering_history = []
        self.transformers = {}
        
        logger.info("FeatureEngineer initialized")
    
    def create_time_series_features(
        self,
        data: pd.DataFrame,
        date_column: str,
        target_columns: Optional[List[str]] = None
    ) -> FeatureEngineeringResult:
        """
        Create time series features
        
        Args:
            data: Input DataFrame
            date_column: Name of date column
            target_columns: Target columns for feature creation
            
        Returns:
            FeatureEngineeringResult object
        """
        try:
            with self.performance_monitor.track_operation("time_series_feature_engineering"):
                if target_columns is None:
                    target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                
                # Ensure date column is datetime
                data_copy = data.copy()
                data_copy[date_column] = pd.to_datetime(data_copy[date_column])
                data_copy = data_copy.sort_values(date_column)
                
                engineered_features = data_copy.copy()
                feature_types = {}
                transformation_info = {}
                
                # Create time-based features
                engineered_features['year'] = data_copy[date_column].dt.year
                engineered_features['month'] = data_copy[date_column].dt.month
                engineered_features['day'] = data_copy[date_column].dt.day
                engineered_features['day_of_week'] = data_copy[date_column].dt.dayofweek
                engineered_features['quarter'] = data_copy[date_column].dt.quarter
                engineered_features['is_weekend'] = data_copy[date_column].dt.dayofweek.isin([5, 6]).astype(int)
                
                feature_types.update({
                    'year': 'temporal',
                    'month': 'temporal',
                    'day': 'temporal',
                    'day_of_week': 'temporal',
                    'quarter': 'temporal',
                    'is_weekend': 'temporal'
                })
                
                # Create lag features for target columns
                for col in target_columns:
                    if col in data_copy.columns:
                        # Lag features
                        for lag in [1, 3, 7, 14, 30]:
                            lag_col = f'{col}_lag_{lag}'
                            engineered_features[lag_col] = data_copy[col].shift(lag)
                            feature_types[lag_col] = 'lag'
                        
                        # Rolling statistics
                        for window in [7, 14, 30]:
                            # Rolling mean
                            mean_col = f'{col}_rolling_mean_{window}'
                            engineered_features[mean_col] = data_copy[col].rolling(window).mean()
                            feature_types[mean_col] = 'rolling'
                            
                            # Rolling std
                            std_col = f'{col}_rolling_std_{window}'
                            engineered_features[std_col] = data_copy[col].rolling(window).std()
                            feature_types[std_col] = 'rolling'
                            
                            # Rolling min/max
                            min_col = f'{col}_rolling_min_{window}'
                            engineered_features[min_col] = data_copy[col].rolling(window).min()
                            feature_types[min_col] = 'rolling'
                            
                            max_col = f'{col}_rolling_max_{window}'
                            engineered_features[max_col] = data_copy[col].rolling(window).max()
                            feature_types[max_col] = 'rolling'
                        
                        # Difference features
                        diff_col = f'{col}_diff_1'
                        engineered_features[diff_col] = data_copy[col].diff()
                        feature_types[diff_col] = 'difference'
                        
                        # Percentage change
                        pct_col = f'{col}_pct_change'
                        engineered_features[pct_col] = data_copy[col].pct_change()
                        feature_types[pct_col] = 'percentage'
                
                # Remove original date column
                engineered_features = engineered_features.drop(columns=[date_column])
                
                # Handle missing values
                engineered_features = engineered_features.fillna(method='ffill').fillna(method='bfill')
                
                result = FeatureEngineeringResult(
                    engineered_features=engineered_features,
                    feature_names=engineered_features.columns.tolist(),
                    feature_types=feature_types,
                    transformation_info={'method': 'time_series', 'target_columns': target_columns}
                )
                
                # Store in history
                self.engineering_history.append(result)
                
                logger.info(f"Created {len(engineered_features.columns)} time series features")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in time series feature engineering: {str(e)}", e)
            raise
    
    def create_statistical_features(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None
    ) -> FeatureEngineeringResult:
        """
        Create statistical features
        
        Args:
            data: Input DataFrame
            target_columns: Target columns for feature creation
            
        Returns:
            FeatureEngineeringResult object
        """
        try:
            with self.performance_monitor.track_operation("statistical_feature_engineering"):
                if target_columns is None:
                    target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                
                engineered_features = data.copy()
                feature_types = {}
                transformation_info = {}
                
                for col in target_columns:
                    if col in data.columns:
                        # Z-score
                        z_col = f'{col}_zscore'
                        engineered_features[z_col] = (data[col] - data[col].mean()) / data[col].std()
                        feature_types[z_col] = 'statistical'
                        
                        # Quantile features
                        for q in [0.25, 0.5, 0.75]:
                            q_col = f'{col}_quantile_{int(q*100)}'
                            engineered_features[q_col] = data[col].quantile(q)
                            feature_types[q_col] = 'statistical'
                        
                        # Skewness and kurtosis
                        if SCIPY_AVAILABLE:
                            skew_col = f'{col}_skewness'
                            engineered_features[skew_col] = stats.skew(data[col].dropna())
                            feature_types[skew_col] = 'statistical'
                            
                            kurt_col = f'{col}_kurtosis'
                            engineered_features[kurt_col] = stats.kurtosis(data[col].dropna())
                            feature_types[kurt_col] = 'statistical'
                        
                        # Range and IQR
                        range_col = f'{col}_range'
                        engineered_features[range_col] = data[col].max() - data[col].min()
                        feature_types[range_col] = 'statistical'
                        
                        q75, q25 = data[col].quantile([0.75, 0.25])
                        iqr_col = f'{col}_iqr'
                        engineered_features[iqr_col] = q75 - q25
                        feature_types[iqr_col] = 'statistical'
                
                result = FeatureEngineeringResult(
                    engineered_features=engineered_features,
                    feature_names=engineered_features.columns.tolist(),
                    feature_types=feature_types,
                    transformation_info={'method': 'statistical', 'target_columns': target_columns}
                )
                
                # Store in history
                self.engineering_history.append(result)
                
                logger.info(f"Created {len(engineered_features.columns)} statistical features")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in statistical feature engineering: {str(e)}", e)
            raise
    
    def create_interaction_features(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        max_interactions: int = 10
    ) -> FeatureEngineeringResult:
        """
        Create interaction features
        
        Args:
            data: Input DataFrame
            feature_columns: Columns to create interactions for
            max_interactions: Maximum number of interaction features
            
        Returns:
            FeatureEngineeringResult object
        """
        try:
            with self.performance_monitor.track_operation("interaction_feature_engineering"):
                if feature_columns is None:
                    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                
                engineered_features = data.copy()
                feature_types = {}
                transformation_info = {}
                
                interaction_count = 0
                
                # Create pairwise interactions
                for i, col1 in enumerate(feature_columns):
                    if col1 not in data.columns:
                        continue
                        
                    for col2 in feature_columns[i+1:]:
                        if col2 not in data.columns or interaction_count >= max_interactions:
                            continue
                        
                        # Multiplication interaction
                        mult_col = f'{col1}_mult_{col2}'
                        engineered_features[mult_col] = data[col1] * data[col2]
                        feature_types[mult_col] = 'interaction'
                        
                        # Division interaction (with safety check)
                        if (data[col2] != 0).all():
                            div_col = f'{col1}_div_{col2}'
                            engineered_features[div_col] = data[col1] / data[col2]
                            feature_types[div_col] = 'interaction'
                        
                        # Addition interaction
                        add_col = f'{col1}_add_{col2}'
                        engineered_features[add_col] = data[col1] + data[col2]
                        feature_types[add_col] = 'interaction'
                        
                        # Subtraction interaction
                        sub_col = f'{col1}_sub_{col2}'
                        engineered_features[sub_col] = data[col1] - data[col2]
                        feature_types[sub_col] = 'interaction'
                        
                        interaction_count += 1
                        
                        if interaction_count >= max_interactions:
                            break
                    
                    if interaction_count >= max_interactions:
                        break
                
                result = FeatureEngineeringResult(
                    engineered_features=engineered_features,
                    feature_names=engineered_features.columns.tolist(),
                    feature_types=feature_types,
                    transformation_info={'method': 'interaction', 'interaction_count': interaction_count}
                )
                
                # Store in history
                self.engineering_history.append(result)
                
                logger.info(f"Created {interaction_count} interaction features")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in interaction feature engineering: {str(e)}", e)
            raise
    
    def apply_feature_scaling(
        self,
        data: pd.DataFrame,
        method: str = 'standard',
        exclude_columns: Optional[List[str]] = None
    ) -> FeatureEngineeringResult:
        """
        Apply feature scaling
        
        Args:
            data: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            FeatureEngineeringResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for feature scaling")
        
        try:
            with self.performance_monitor.track_operation("feature_scaling"):
                if exclude_columns is None:
                    exclude_columns = []
                
                # Select numeric columns for scaling
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                scale_columns = [col for col in numeric_columns if col not in exclude_columns]
                
                engineered_features = data.copy()
                feature_types = {}
                transformation_info = {}
                
                if method == 'standard':
                    scaler = StandardScaler()
                    feature_types.update({col: 'scaled_standard' for col in scale_columns})
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                    feature_types.update({col: 'scaled_minmax' for col in scale_columns})
                elif method == 'robust':
                    scaler = RobustScaler()
                    feature_types.update({col: 'scaled_robust' for col in scale_columns})
                else:
                    raise ValueError(f"Unsupported scaling method: {method}")
                
                # Apply scaling
                if scale_columns:
                    engineered_features[scale_columns] = scaler.fit_transform(data[scale_columns])
                    
                    # Store scaler
                    self.transformers[f'scaler_{method}'] = scaler
                    
                    transformation_info = {
                        'method': f'scaling_{method}',
                        'scaled_columns': scale_columns,
                        'scaler_params': scaler.get_params()
                    }
                
                result = FeatureEngineeringResult(
                    engineered_features=engineered_features,
                    feature_names=engineered_features.columns.tolist(),
                    feature_types=feature_types,
                    transformation_info=transformation_info
                )
                
                # Store in history
                self.engineering_history.append(result)
                
                logger.info(f"Applied {method} scaling to {len(scale_columns)} features")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in feature scaling: {str(e)}", e)
            raise
    
    def apply_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        method: str = 'pca',
        n_components: Optional[int] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> FeatureEngineeringResult:
        """
        Apply dimensionality reduction
        
        Args:
            data: Input DataFrame
            method: Reduction method ('pca', 'feature_selection')
            n_components: Number of components to keep
            exclude_columns: Columns to exclude from reduction
            
        Returns:
            FeatureEngineeringResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for dimensionality reduction")
        
        try:
            with self.performance_monitor.track_operation("dimensionality_reduction"):
                if exclude_columns is None:
                    exclude_columns = []
                
                # Select numeric columns for reduction
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                reduce_columns = [col for col in numeric_columns if col not in exclude_columns]
                
                if n_components is None:
                    n_components = min(10, len(reduce_columns))
                
                engineered_features = data.copy()
                feature_types = {}
                transformation_info = {}
                
                if method == 'pca':
                    # PCA reduction
                    pca = PCA(n_components=n_components)
                    pca_features = pca.fit_transform(data[reduce_columns])
                    
                    # Create new feature names
                    pca_feature_names = [f'pca_component_{i+1}' for i in range(n_components)]
                    
                    # Add PCA features to DataFrame
                    for i, name in enumerate(pca_feature_names):
                        engineered_features[name] = pca_features[:, i]
                        feature_types[name] = 'pca'
                    
                    # Store PCA transformer
                    self.transformers['pca'] = pca
                    
                    transformation_info = {
                        'method': 'pca',
                        'n_components': n_components,
                        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                        'original_columns': reduce_columns
                    }
                    
                elif method == 'feature_selection':
                    # Feature selection
                    selector = SelectKBest(score_func=f_regression, k=n_components)
                    selected_features = selector.fit_transform(data[reduce_columns], data[reduce_columns[0]])
                    
                    # Get selected feature names
                    selected_indices = selector.get_support()
                    selected_names = [reduce_columns[i] for i in range(len(reduce_columns)) if selected_indices[i]]
                    
                    # Keep only selected features
                    engineered_features = engineered_features[selected_names + exclude_columns]
                    feature_types.update({col: 'selected' for col in selected_names})
                    
                    # Store selector
                    self.transformers['feature_selector'] = selector
                    
                    transformation_info = {
                        'method': 'feature_selection',
                        'n_components': n_components,
                        'selected_columns': selected_names,
                        'scores': selector.scores_.tolist()
                    }
                    
                else:
                    raise ValueError(f"Unsupported reduction method: {method}")
                
                result = FeatureEngineeringResult(
                    engineered_features=engineered_features,
                    feature_names=engineered_features.columns.tolist(),
                    feature_types=feature_types,
                    transformation_info=transformation_info
                )
                
                # Store in history
                self.engineering_history.append(result)
                
                logger.info(f"Applied {method} reduction to {len(reduce_columns)} features")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in dimensionality reduction: {str(e)}", e)
            raise
    
    def remove_correlated_features(
        self,
        data: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> FeatureEngineeringResult:
        """
        Remove highly correlated features
        
        Args:
            data: Input DataFrame
            threshold: Correlation threshold for removal
            
        Returns:
            FeatureEngineeringResult object
        """
        try:
            with self.performance_monitor.track_operation("correlation_feature_removal"):
                if threshold is None:
                    threshold = self.correlation_threshold
                
                # Select numeric columns
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                
                # Calculate correlation matrix
                corr_matrix = data[numeric_columns].corr().abs()
                
                # Find highly correlated features
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
                
                # Remove correlated features
                engineered_features = data.drop(columns=to_drop)
                feature_types = {col: 'uncorrelated' for col in engineered_features.columns}
                
                transformation_info = {
                    'method': 'correlation_removal',
                    'threshold': threshold,
                    'removed_features': to_drop,
                    'remaining_features': len(engineered_features.columns)
                }
                
                result = FeatureEngineeringResult(
                    engineered_features=engineered_features,
                    feature_names=engineered_features.columns.tolist(),
                    feature_types=feature_types,
                    transformation_info=transformation_info
                )
                
                # Store in history
                self.engineering_history.append(result)
                
                logger.info(f"Removed {len(to_drop)} correlated features")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in correlation feature removal: {str(e)}", e)
            raise
    
    def get_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering results"""
        if not self.engineering_history:
            return {"message": "No feature engineering results found"}
        
        summary = {
            'total_operations': len(self.engineering_history),
            'methods_used': list(set(result.transformation_info.get('method', 'unknown') 
                                   for result in self.engineering_history)),
            'total_features_created': sum(len(result.feature_names) 
                                        for result in self.engineering_history),
            'feature_type_distribution': {}
        }
        
        # Calculate feature type distribution
        all_feature_types = []
        for result in self.engineering_history:
            all_feature_types.extend(result.feature_types.values())
        
        if all_feature_types:
            type_counts = {}
            for feature_type in set(all_feature_types):
                type_counts[feature_type] = all_feature_types.count(feature_type)
            summary['feature_type_distribution'] = type_counts
        
        return summary
    
    def export_engineering_results(self, filepath: str) -> None:
        """Export feature engineering results"""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                'engineering_history': [
                    {
                        'method': result.transformation_info.get('method', 'unknown'),
                        'feature_count': len(result.feature_names),
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                        'transformation_info': result.transformation_info
                    }
                    for result in self.engineering_history
                ],
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Feature engineering results exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting engineering results: {str(e)}", e)
