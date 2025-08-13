"""
Deep Learning Engine

This module provides deep learning capabilities including neural networks,
multi-layer perceptrons, CNNs, LSTMs, and transformers.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime

# Try to import deep learning frameworks
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

from src.core.error_handling_service import ErrorHandlingService
from src.config.advanced_ml_config import get_advanced_ml_config

logger = logging.getLogger(__name__)
error_handler = ErrorHandlingService()


class DeepLearningEngine:
    """Deep learning engine supporting multiple frameworks and architectures."""
    
    def __init__(self):
        self.config = get_advanced_ml_config()
        self.framework = self.config.deep_learning.framework
        self.models = {}
        self.history = {}
        
        # Validate framework availability
        if self.framework == "tensorflow" and not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, falling back to PyTorch")
            self.framework = "pytorch"
        
        if self.framework == "pytorch" and not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to TensorFlow")
            self.framework = "tensorflow"
        
        if not TENSORFLOW_AVAILABLE and not PYTORCH_AVAILABLE:
            raise ImportError("Neither TensorFlow nor PyTorch is available")
        
        logger.info(f"Initialized DeepLearningEngine with {self.framework}")
    
    def create_mlp(self, input_dim: int, output_dim: int, 
                   architecture: str = "mlp") -> Any:
        """Create a Multi-Layer Perceptron."""
        try:
            config = self.config.deep_learning.architectures[architecture]
            
            if self.framework == "tensorflow":
                return self._create_tensorflow_mlp(input_dim, output_dim, config)
            else:
                return self._create_pytorch_mlp(input_dim, output_dim, config)
                
        except Exception as e:
            error_handler.handle_error(f"Error creating MLP: {str(e)}", e)
            return None
    
    def _create_tensorflow_mlp(self, input_dim: int, output_dim: int, 
                              config: Dict[str, Any]) -> Any:
        """Create TensorFlow MLP model."""
        if not TENSORFLOW_AVAILABLE or keras is None:
            raise ImportError("TensorFlow/Keras not available")
            
        layers = config.get("layers", [512, 256, 128, 64])
        activation = config.get("activation", "relu")
        dropout = config.get("dropout", 0.3)
        batch_norm = config.get("batch_normalization", True)
        
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Dense(layers[0], input_dim=input_dim))
        if batch_norm:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.Dropout(dropout))
        
        # Hidden layers
        for units in layers[1:]:
            model.add(keras.layers.Dense(units))
            if batch_norm:
                model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation(activation))
            model.add(keras.layers.Dropout(dropout))
        
        # Output layer
        if output_dim == 1:
            model.add(keras.layers.Dense(1, activation="sigmoid"))
        else:
            model.add(keras.layers.Dense(output_dim, activation="softmax"))
        
        return model
    
    def _create_pytorch_mlp(self, input_dim: int, output_dim: int, 
                           config: Dict[str, Any]) -> Any:
        """Create PyTorch MLP model."""
        if not PYTORCH_AVAILABLE or nn is None:
            raise ImportError("PyTorch not available")
            
        layers = config.get("layers", [512, 256, 128, 64])
        activation = config.get("activation", "relu")
        dropout = config.get("dropout", 0.3)
        batch_norm = config.get("batch_normalization", True)
        
        class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                self.layers = nn.ModuleList()
                
                # Input layer
                self.layers.append(nn.Linear(input_dim, layers[0]))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(layers[0]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
                
                # Hidden layers
                for i in range(len(layers) - 1):
                    self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                    if batch_norm:
                        self.layers.append(nn.BatchNorm1d(layers[i + 1]))
                    self.layers.append(nn.ReLU())
                    self.layers.append(nn.Dropout(dropout))
                
                # Output layer
                if output_dim == 1:
                    self.layers.append(nn.Linear(layers[-1], 1))
                    self.layers.append(nn.Sigmoid())
                else:
                    self.layers.append(nn.Linear(layers[-1], output_dim))
                    self.layers.append(nn.Softmax(dim=1))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return MLP()
    
    def create_cnn(self, input_shape: Tuple[int, ...], 
                   output_dim: int) -> Any:
        """Create a Convolutional Neural Network."""
        try:
            config = self.config.deep_learning.architectures["cnn"]
            
            if self.framework == "tensorflow":
                return self._create_tensorflow_cnn(input_shape, output_dim, config)
            else:
                return self._create_pytorch_cnn(input_shape, output_dim, config)
                
        except Exception as e:
            error_handler.handle_error(f"Error creating CNN: {str(e)}", e)
            return None
    
    def _create_tensorflow_cnn(self, input_shape: Tuple[int, ...], 
                              output_dim: int, config: Dict[str, Any]) -> Any:
        """Create TensorFlow CNN model."""
        filters = config.get("filters", [32, 64, 128, 256])
        kernel_sizes = config.get("kernel_sizes", [3, 3, 3, 3])
        pooling = config.get("pooling", "max")
        dropout = config.get("dropout", 0.5)
        
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Input(shape=input_shape))
        
        # Convolutional layers
        for i, (filters_count, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            model.add(keras.layers.Conv2D(filters_count, kernel_size, padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            
            if pooling == "max":
                model.add(keras.layers.MaxPooling2D(2, 2))
            else:
                model.add(keras.layers.AveragePooling2D(2, 2))
            
            model.add(keras.layers.Dropout(dropout))
        
        # Flatten and dense layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(dropout))
        
        # Output layer
        if output_dim == 1:
            model.add(keras.layers.Dense(1, activation="sigmoid"))
        else:
            model.add(keras.layers.Dense(output_dim, activation="softmax"))
        
        return model
    
    def _create_pytorch_cnn(self, input_shape: Tuple[int, ...], 
                           output_dim: int, config: Dict[str, Any]) -> Any:
        """Create PyTorch CNN model."""
        filters = config.get("filters", [32, 64, 128, 256])
        kernel_sizes = config.get("kernel_sizes", [3, 3, 3, 3])
        pooling = config.get("pooling", "max")
        dropout = config.get("dropout", 0.5)
        
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv_layers = nn.ModuleList()
                self.pool_layers = nn.ModuleList()
                
                # Convolutional layers
                in_channels = input_shape[0] if len(input_shape) == 3 else 1
                for filters_count, kernel_size in zip(filters, kernel_sizes):
                    self.conv_layers.append(
                        nn.Conv2d(in_channels, filters_count, kernel_size, padding=1)
                    )
                    self.conv_layers.append(nn.BatchNorm2d(filters_count))
                    self.conv_layers.append(nn.ReLU())
                    
                    if pooling == "max":
                        self.pool_layers.append(nn.MaxPool2d(2, 2))
                    else:
                        self.pool_layers.append(nn.AvgPool2d(2, 2))
                    
                    self.conv_layers.append(nn.Dropout2d(dropout))
                    in_channels = filters_count
                
                # Calculate flattened size
                # This is a simplified calculation - in practice, you'd need to compute this
                flattened_size = filters[-1] * 4 * 4  # Approximate
                
                self.fc_layers = nn.Sequential(
                    nn.Linear(flattened_size, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, output_dim)
                )
                
                if output_dim == 1:
                    self.fc_layers.add_module("sigmoid", nn.Sigmoid())
                else:
                    self.fc_layers.add_module("softmax", nn.Softmax(dim=1))
            
            def forward(self, x):
                # Apply conv layers
                for i, layer in enumerate(self.conv_layers):
                    x = layer(x)
                    if isinstance(layer, nn.Dropout2d):
                        x = self.pool_layers[i // 4](x)
                
                x = x.view(x.size(0), -1)  # Flatten
                x = self.fc_layers(x)
                return x
        
        return CNN()
    
    def create_lstm(self, input_shape: Tuple[int, ...], 
                    output_dim: int) -> Any:
        """Create an LSTM model for sequence data."""
        try:
            config = self.config.deep_learning.architectures["lstm"]
            
            if self.framework == "tensorflow":
                return self._create_tensorflow_lstm(input_shape, output_dim, config)
            else:
                return self._create_pytorch_lstm(input_shape, output_dim, config)
                
        except Exception as e:
            error_handler.handle_error(f"Error creating LSTM: {str(e)}", e)
            return None
    
    def _create_tensorflow_lstm(self, input_shape: Tuple[int, ...], 
                               output_dim: int, config: Dict[str, Any]) -> Any:
        """Create TensorFlow LSTM model."""
        units = config.get("units", [128, 64, 32])
        dropout = config.get("dropout", 0.2)
        recurrent_dropout = config.get("recurrent_dropout", 0.2)
        bidirectional = config.get("bidirectional", True)
        
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Input(shape=input_shape))
        
        # LSTM layers
        for i, unit in enumerate(units):
            if bidirectional:
                model.add(keras.layers.Bidirectional(
                    keras.layers.LSTM(unit, return_sequences=i < len(units) - 1,
                                    dropout=dropout, recurrent_dropout=recurrent_dropout)
                ))
            else:
                model.add(keras.layers.LSTM(unit, return_sequences=i < len(units) - 1,
                                          dropout=dropout, recurrent_dropout=recurrent_dropout))
        
        # Output layer
        if output_dim == 1:
            model.add(keras.layers.Dense(1, activation="sigmoid"))
        else:
            model.add(keras.layers.Dense(output_dim, activation="softmax"))
        
        return model
    
    def _create_pytorch_lstm(self, input_shape: Tuple[int, ...], 
                            output_dim: int, config: Dict[str, Any]) -> nn.Module:
        """Create PyTorch LSTM model."""
        units = config.get("units", [128, 64, 32])
        dropout = config.get("dropout", 0.2)
        bidirectional = config.get("bidirectional", True)
        
        class LSTM(nn.Module):
            def __init__(self):
                super(LSTM, self).__init__()
                self.lstm_layers = nn.ModuleList()
                
                input_size = input_shape[-1]
                for i, hidden_size in enumerate(units):
                    self.lstm_layers.append(
                        nn.LSTM(input_size, hidden_size, batch_first=True,
                               dropout=dropout if i < len(units) - 1 else 0,
                               bidirectional=bidirectional)
                    )
                    input_size = hidden_size * 2 if bidirectional else hidden_size
                
                self.output_layer = nn.Linear(input_size, output_dim)
                if output_dim == 1:
                    self.activation = nn.Sigmoid()
                else:
                    self.activation = nn.Softmax(dim=1)
            
            def forward(self, x):
                for lstm_layer in self.lstm_layers:
                    x, _ = lstm_layer(x)
                
                # Take the last output
                x = x[:, -1, :]
                x = self.output_layer(x)
                x = self.activation(x)
                return x
        
        return LSTM()
    
    def train_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, 
                   y_val: Optional[np.ndarray] = None,
                   model_name: str = "model") -> Dict[str, Any]:
        """Train a deep learning model."""
        try:
            training_config = self.config.deep_learning.training
            
            if self.framework == "tensorflow":
                return self._train_tensorflow_model(model, X_train, y_train, 
                                                  X_val, y_val, training_config, model_name)
            else:
                return self._train_pytorch_model(model, X_train, y_train, 
                                               X_val, y_val, training_config, model_name)
                
        except Exception as e:
            error_handler.handle_error(f"Error training model: {str(e)}", e)
            return {}
    
    def _train_tensorflow_model(self, model: Any, X_train: np.ndarray, 
                               y_train: np.ndarray, X_val: Optional[np.ndarray],
                               y_val: Optional[np.ndarray], config: Dict[str, Any],
                               model_name: str) -> Dict[str, Any]:
        """Train TensorFlow model."""
        # Compile model
        optimizer = config.get("optimizer", "adam")
        loss_function = config.get("loss_function", "categorical_crossentropy")
        learning_rate = config.get("learning_rate", 0.001)
        
        if optimizer == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        
        # Callbacks
        callbacks = []
        if config.get("early_stopping", True):
            callbacks.append(keras.callbacks.EarlyStopping(
                patience=config.get("patience", 10),
                restore_best_weights=True
            ))
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=config.get("batch_size", 32),
            epochs=config.get("epochs", 100),
            validation_data=(X_val, y_val) if X_val is not None else None,
            validation_split=config.get("validation_split", 0.2) if X_val is None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model and history
        self.models[model_name] = model
        self.history[model_name] = history.history
        
        return {
            "model": model,
            "history": history.history,
            "framework": "tensorflow"
        }
    
    def _train_pytorch_model(self, model: nn.Module, X_train: np.ndarray,
                            y_train: np.ndarray, X_val: Optional[np.ndarray],
                            y_val: Optional[np.ndarray], config: Dict[str, Any],
                            model_name: str) -> Dict[str, Any]:
        """Train PyTorch model."""
        # Convert numpy arrays to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), 
                              lr=config.get("learning_rate", 0.001))
        
        if y_train.shape[1] == 1:  # Binary classification
            criterion = nn.BCELoss()
        else:  # Multi-class classification
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 100)
        
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == torch.max(batch_y, 1)[1]).sum().item()
            
            # Validation
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_correct = (val_predicted == torch.max(y_val_tensor, 1)[1]).sum().item()
                    val_accuracy = val_correct / len(y_val_tensor)
            
            # Record history
            history["loss"].append(total_loss / (len(X_train_tensor) // batch_size))
            history["accuracy"].append(correct / total)
            
            if X_val is not None:
                history["val_loss"].append(val_loss.item())
                history["val_accuracy"].append(val_accuracy)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={history['loss'][-1]:.4f}, "
                          f"Accuracy={history['accuracy'][-1]:.4f}")
        
        # Store model and history
        self.models[model_name] = model
        self.history[model_name] = history
        
        return {
            "model": model,
            "history": history,
            "framework": "pytorch"
        }
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        try:
            if self.framework == "tensorflow":
                return model.predict(X)
            else:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    predictions = model(X_tensor)
                    return predictions.numpy()
                    
        except Exception as e:
            error_handler.handle_error(f"Error making predictions: {str(e)}", e)
            return np.array([])
    
    def save_model(self, model: Any, model_name: str, 
                  save_path: Optional[str] = None) -> bool:
        """Save a trained model."""
        try:
            if save_path is None:
                save_path = self.config.deep_learning.model_storage["base_path"]
            
            os.makedirs(save_path, exist_ok=True)
            
            if self.framework == "tensorflow":
                model_path = os.path.join(save_path, f"{model_name}.h5")
                model.save(model_path)
            else:
                model_path = os.path.join(save_path, f"{model_name}.pth")
                torch.save(model.state_dict(), model_path)
            
            # Save metadata
            metadata = {
                "framework": self.framework,
                "model_name": model_name,
                "created_at": datetime.now().isoformat(),
                "architecture": "deep_learning"
            }
            
            metadata_path = os.path.join(save_path, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            error_handler.handle_error(f"Error saving model: {str(e)}", e)
            return False
    
    def load_model(self, model_name: str, 
                  load_path: Optional[str] = None) -> Optional[Any]:
        """Load a saved model."""
        try:
            if load_path is None:
                load_path = self.config.deep_learning.model_storage["base_path"]
            
            # Load metadata
            metadata_path = os.path.join(load_path, f"{model_name}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    framework = metadata.get("framework", self.framework)
            else:
                framework = self.framework
            
            if framework == "tensorflow":
                model_path = os.path.join(load_path, f"{model_name}.h5")
                if os.path.exists(model_path):
                    return keras.models.load_model(model_path)
            else:
                model_path = os.path.join(load_path, f"{model_name}.pth")
                if os.path.exists(model_path):
                    # Note: This is a simplified loading - in practice you'd need
                    # to recreate the model architecture first
                    return torch.load(model_path)
            
            logger.warning(f"Model {model_name} not found at {load_path}")
            return None
            
        except Exception as e:
            error_handler.handle_error(f"Error loading model: {str(e)}", e)
            return None
    
    def get_model_summary(self, model: Any) -> str:
        """Get a summary of the model architecture."""
        try:
            if self.framework == "tensorflow":
                # Capture the summary output
                from io import StringIO
                summary_io = StringIO()
                model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
                return summary_io.getvalue()
            else:
                # For PyTorch, return a simple parameter count
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                return f"PyTorch Model\nTotal parameters: {total_params:,}\nTrainable parameters: {trainable_params:,}"
                
        except Exception as e:
            error_handler.handle_error(f"Error getting model summary: {str(e)}", e)
            return "Error getting model summary"
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            predictions = self.predict(model, X_test)
            
            # Calculate metrics
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                # Binary classification
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                y_pred = (predictions > 0.5).astype(int)
                y_true = y_test.astype(int)
            else:
                # Multi-class classification
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                y_pred = np.argmax(predictions, axis=1)
                y_true = np.argmax(y_test, axis=1)
            
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "f1_score": f1_score(y_true, y_pred, average='weighted')
            }
            
            return metrics
            
        except Exception as e:
            error_handler.handle_error(f"Error evaluating model: {str(e)}", e)
            return {}
