"""
Transfer Learning Service

This module provides transfer learning capabilities for adapting pre-trained
models to new domains and tasks.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
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

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None
    AutoModelForSequenceClassification = None

from src.core.error_handling_service import ErrorHandlingService
from src.config.advanced_ml_config import get_advanced_ml_config

logger = logging.getLogger(__name__)
error_handler = ErrorHandlingService()


class TransferLearningService:
    """Service for transfer learning and pre-trained model adaptation."""
    
    def __init__(self):
        self.config = get_advanced_ml_config()
        self.framework = self.config.deep_learning.framework
        self.pre_trained_models = {}
        self.fine_tuned_models = {}
        
        # Validate framework availability
        if self.framework == "tensorflow" and not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, falling back to PyTorch")
            self.framework = "pytorch"
        
        if self.framework == "pytorch" and not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to TensorFlow")
            self.framework = "tensorflow"
        
        if not TENSORFLOW_AVAILABLE and not PYTORCH_AVAILABLE:
            raise ImportError("Neither TensorFlow nor PyTorch is available")
        
        logger.info(f"Initialized TransferLearningService with {self.framework}")
    
    def load_pre_trained_model(self, model_type: str, 
                              model_name: Optional[str] = None) -> Any:
        """Load a pre-trained model."""
        try:
            if model_name is None:
                model_name = self.config.transfer_learning.pre_trained_models[model_type]["model_name"]
            
            if model_type in ["bert", "gpt2"] and TRANSFORMERS_AVAILABLE:
                return self._load_transformers_model(model_type, model_name)
            elif model_type in ["resnet", "vgg"]:
                if self.framework == "tensorflow":
                    return self._load_tensorflow_vision_model(model_type, model_name)
                else:
                    return self._load_pytorch_vision_model(model_type, model_name)
            else:
                logger.warning(f"Unsupported model type: {model_type}")
                return None
                
        except Exception as e:
            error_handler.handle_error(f"Error loading pre-trained model: {str(e)}", e)
            return None
    
    def _load_transformers_model(self, model_type: str, model_name: str) -> Any:
        """Load a transformers model."""
        try:
            if model_type == "bert":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
            elif model_type == "gpt2":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
            else:
                logger.warning(f"Unsupported transformers model type: {model_type}")
                return None
            
            self.pre_trained_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": model_type
            }
            
            logger.info(f"Loaded {model_type} model: {model_name}")
            return self.pre_trained_models[model_name]
            
        except Exception as e:
            error_handler.handle_error(f"Error loading transformers model: {str(e)}", e)
            return None
    
    def _load_tensorflow_vision_model(self, model_type: str, model_name: str) -> Any:
        """Load a TensorFlow vision model."""
        try:
            if model_type == "resnet":
                model = keras.applications.ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
            elif model_type == "vgg":
                model = keras.applications.VGG16(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
            else:
                logger.warning(f"Unsupported TensorFlow vision model type: {model_type}")
                return None
            
            self.pre_trained_models[model_name] = {
                "model": model,
                "type": model_type,
                "framework": "tensorflow"
            }
            
            logger.info(f"Loaded TensorFlow {model_type} model: {model_name}")
            return self.pre_trained_models[model_name]
            
        except Exception as e:
            error_handler.handle_error(f"Error loading TensorFlow vision model: {str(e)}", e)
            return None
    
    def _load_pytorch_vision_model(self, model_type: str, model_name: str) -> Any:
        """Load a PyTorch vision model."""
        try:
            import torchvision.models as models
            
            if model_type == "resnet":
                model = models.resnet50(pretrained=True)
                # Remove the final classification layer
                model = nn.Sequential(*list(model.children())[:-1])
            elif model_type == "vgg":
                model = models.vgg16(pretrained=True)
                # Remove the final classification layer
                model = nn.Sequential(*list(model.children())[:-1])
            else:
                logger.warning(f"Unsupported PyTorch vision model type: {model_type}")
                return None
            
            self.pre_trained_models[model_name] = {
                "model": model,
                "type": model_type,
                "framework": "pytorch"
            }
            
            logger.info(f"Loaded PyTorch {model_type} model: {model_name}")
            return self.pre_trained_models[model_name]
            
        except Exception as e:
            error_handler.handle_error(f"Error loading PyTorch vision model: {str(e)}", e)
            return None
    
    def create_fine_tuned_model(self, base_model_name: str, 
                               output_dim: int, task_type: str = "classification") -> Any:
        """Create a fine-tuned model from a pre-trained model."""
        try:
            if base_model_name not in self.pre_trained_models:
                logger.error(f"Pre-trained model {base_model_name} not found")
                return None
            
            base_model_info = self.pre_trained_models[base_model_name]
            model_type = base_model_info["type"]
            
            if model_type in ["bert", "gpt2"]:
                return self._create_fine_tuned_transformers_model(
                    base_model_info, output_dim, task_type
                )
            elif model_type in ["resnet", "vgg"]:
                return self._create_fine_tuned_vision_model(
                    base_model_info, output_dim, task_type
                )
            else:
                logger.warning(f"Unsupported model type for fine-tuning: {model_type}")
                return None
                
        except Exception as e:
            error_handler.handle_error(f"Error creating fine-tuned model: {str(e)}", e)
            return None
    
    def _create_fine_tuned_transformers_model(self, base_model_info: Dict[str, Any],
                                            output_dim: int, task_type: str) -> Any:
        """Create a fine-tuned transformers model."""
        try:
            model = base_model_info["model"]
            tokenizer = base_model_info["tokenizer"]
            
            # For classification tasks, we need to add a classification head
            if task_type == "classification":
                if hasattr(model, 'config'):
                    model.config.num_labels = output_dim
                
                # Create a simple classification head
                if self.framework == "tensorflow":
                    # This is a simplified approach - in practice you'd use
                    # AutoModelForSequenceClassification
                    classification_head = keras.Sequential([
                        keras.layers.Dense(512, activation='relu'),
                        keras.layers.Dropout(0.3),
                        keras.layers.Dense(output_dim, activation='softmax')
                    ])
                    
                    # Combine base model with classification head
                    fine_tuned_model = keras.Sequential([
                        model,
                        keras.layers.GlobalAveragePooling1D(),
                        classification_head
                    ])
                else:
                    # PyTorch approach
                    class FineTunedModel(nn.Module):
                        def __init__(self, base_model, output_dim):
                            super(FineTunedModel, self).__init__()
                            self.base_model = base_model
                            self.classification_head = nn.Sequential(
                                nn.Linear(768, 512),  # Assuming BERT hidden size
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(512, output_dim),
                                nn.Softmax(dim=1)
                            )
                        
                        def forward(self, input_ids, attention_mask=None):
                            outputs = self.base_model(input_ids, attention_mask=attention_mask)
                            pooled_output = outputs.pooler_output
                            return self.classification_head(pooled_output)
                    
                    fine_tuned_model = FineTunedModel(model, output_dim)
            
            else:
                # For other tasks, return the base model
                fine_tuned_model = model
            
            return {
                "model": fine_tuned_model,
                "tokenizer": tokenizer,
                "base_model_info": base_model_info
            }
            
        except Exception as e:
            error_handler.handle_error(f"Error creating fine-tuned transformers model: {str(e)}", e)
            return None
    
    def _create_fine_tuned_vision_model(self, base_model_info: Dict[str, Any],
                                      output_dim: int, task_type: str) -> Any:
        """Create a fine-tuned vision model."""
        try:
            base_model = base_model_info["model"]
            framework = base_model_info["framework"]
            
            if framework == "tensorflow":
                # Freeze the base model layers
                base_model.trainable = False
                
                # Add classification head
                classification_head = keras.Sequential([
                    keras.layers.GlobalAveragePooling2D(),
                    keras.layers.Dense(512, activation='relu'),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(output_dim, activation='softmax')
                ])
                
                fine_tuned_model = keras.Sequential([
                    base_model,
                    classification_head
                ])
                
            else:
                # PyTorch approach
                class FineTunedVisionModel(nn.Module):
                    def __init__(self, base_model, output_dim):
                        super(FineTunedVisionModel, self).__init__()
                        self.base_model = base_model
                        self.classification_head = nn.Sequential(
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(),
                            nn.Linear(2048, 512),  # Assuming ResNet feature size
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, output_dim),
                            nn.Softmax(dim=1)
                        )
                    
                    def forward(self, x):
                        x = self.base_model(x)
                        return self.classification_head(x)
                
                fine_tuned_model = FineTunedVisionModel(base_model, output_dim)
            
            return {
                "model": fine_tuned_model,
                "base_model_info": base_model_info
            }
            
        except Exception as e:
            error_handler.handle_error(f"Error creating fine-tuned vision model: {str(e)}", e)
            return None
    
    def fine_tune_model(self, fine_tuned_model: Any, X_train: np.ndarray, 
                       y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None, model_name: str = "fine_tuned") -> Dict[str, Any]:
        """Fine-tune a pre-trained model."""
        try:
            settings = self.config.transfer_learning.settings
            
            if self.framework == "tensorflow":
                return self._fine_tune_tensorflow_model(
                    fine_tuned_model, X_train, y_train, X_val, y_val, settings, model_name
                )
            else:
                return self._fine_tune_pytorch_model(
                    fine_tuned_model, X_train, y_train, X_val, y_val, settings, model_name
                )
                
        except Exception as e:
            error_handler.handle_error(f"Error fine-tuning model: {str(e)}", e)
            return {}
    
    def _fine_tune_tensorflow_model(self, fine_tuned_model: Any, X_train: np.ndarray,
                                  y_train: np.ndarray, X_val: Optional[np.ndarray],
                                  y_val: Optional[np.ndarray], settings: Dict[str, Any],
                                  model_name: str) -> Dict[str, Any]:
        """Fine-tune a TensorFlow model."""
        try:
            # Phase 1: Train only the classification head
            fine_tuned_model["model"].trainable = False
            
            optimizer = keras.optimizers.Adam(
                learning_rate=settings.get("learning_rate_multiplier", 0.1) * 0.001
            )
            
            fine_tuned_model["model"].compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train classification head
            history_head = fine_tuned_model["model"].fit(
                X_train, y_train,
                batch_size=32,
                epochs=settings.get("fine_tune_epochs", 50) // 2,
                validation_data=(X_val, y_val) if X_val is not None else None,
                validation_split=0.2 if X_val is None else None,
                verbose=1
            )
            
            # Phase 2: Gradual unfreezing if enabled
            if settings.get("gradual_unfreezing", True):
                # Unfreeze some layers and continue training
                fine_tuned_model["model"].trainable = True
                
                # Set different learning rates for different layers
                optimizer = keras.optimizers.Adam(
                    learning_rate=settings.get("learning_rate_multiplier", 0.1) * 0.0001
                )
                
                fine_tuned_model["model"].compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Continue training
                history_full = fine_tuned_model["model"].fit(
                    X_train, y_train,
                    batch_size=32,
                    epochs=settings.get("fine_tune_epochs", 50) // 2,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    validation_split=0.2 if X_val is None else None,
                    verbose=1
                )
                
                # Combine histories
                history = {
                    "loss": history_head.history["loss"] + history_full.history["loss"],
                    "accuracy": history_head.history["accuracy"] + history_full.history["accuracy"],
                    "val_loss": history_head.history.get("val_loss", []) + history_full.history.get("val_loss", []),
                    "val_accuracy": history_head.history.get("val_accuracy", []) + history_full.history.get("val_accuracy", [])
                }
            else:
                history = history_head.history
            
            # Store model and history
            self.fine_tuned_models[model_name] = fine_tuned_model
            self.fine_tuned_models[model_name]["history"] = history
            
            return {
                "model": fine_tuned_model["model"],
                "history": history,
                "framework": "tensorflow"
            }
            
        except Exception as e:
            error_handler.handle_error(f"Error fine-tuning TensorFlow model: {str(e)}", e)
            return {}
    
    def _fine_tune_pytorch_model(self, fine_tuned_model: Any, X_train: np.ndarray,
                               y_train: np.ndarray, X_val: Optional[np.ndarray],
                               y_val: Optional[np.ndarray], settings: Dict[str, Any],
                               model_name: str) -> Dict[str, Any]:
        """Fine-tune a PyTorch model."""
        try:
            model = fine_tuned_model["model"]
            
            # Convert numpy arrays to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            
            if X_val is not None:
                X_val_tensor = torch.FloatTensor(X_val)
                y_val_tensor = torch.FloatTensor(y_val)
            
            # Setup optimizer and loss function
            optimizer = optim.Adam(
                model.parameters(),
                lr=settings.get("learning_rate_multiplier", 0.1) * 0.001
            )
            
            if y_train.shape[1] == 1:  # Binary classification
                criterion = nn.BCELoss()
            else:  # Multi-class classification
                criterion = nn.CrossEntropyLoss()
            
            # Training loop
            epochs = settings.get("fine_tune_epochs", 50)
            history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                # Training
                for i in range(0, len(X_train_tensor), 32):
                    batch_X = X_train_tensor[i:i+32]
                    batch_y = y_train_tensor[i:i+32]
                    
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
                history["loss"].append(total_loss / (len(X_train_tensor) // 32))
                history["accuracy"].append(correct / total)
                
                if X_val is not None:
                    history["val_loss"].append(val_loss.item())
                    history["val_accuracy"].append(val_accuracy)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss={history['loss'][-1]:.4f}, "
                              f"Accuracy={history['accuracy'][-1]:.4f}")
            
            # Store model and history
            self.fine_tuned_models[model_name] = fine_tuned_model
            self.fine_tuned_models[model_name]["history"] = history
            
            return {
                "model": model,
                "history": history,
                "framework": "pytorch"
            }
            
        except Exception as e:
            error_handler.handle_error(f"Error fine-tuning PyTorch model: {str(e)}", e)
            return {}
    
    def predict(self, model: Any, X: np.ndarray, 
               model_type: str = "transformers") -> np.ndarray:
        """Make predictions using a fine-tuned model."""
        try:
            if model_type == "transformers":
                return self._predict_transformers(model, X)
            else:
                return self._predict_vision(model, X)
                
        except Exception as e:
            error_handler.handle_error(f"Error making predictions: {str(e)}", e)
            return np.array([])
    
    def _predict_transformers(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using a transformers model."""
        try:
            if self.framework == "tensorflow":
                return model["model"].predict(X)
            else:
                model["model"].eval()
                with torch.no_grad():
                    # This is a simplified approach - in practice you'd need to tokenize the input
                    X_tensor = torch.FloatTensor(X)
                    predictions = model["model"](X_tensor)
                    return predictions.numpy()
                    
        except Exception as e:
            error_handler.handle_error(f"Error making transformers predictions: {str(e)}", e)
            return np.array([])
    
    def _predict_vision(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using a vision model."""
        try:
            if self.framework == "tensorflow":
                return model["model"].predict(X)
            else:
                model["model"].eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    predictions = model["model"](X_tensor)
                    return predictions.numpy()
                    
        except Exception as e:
            error_handler.handle_error(f"Error making vision predictions: {str(e)}", e)
            return np.array([])
    
    def save_fine_tuned_model(self, model: Any, model_name: str,
                            save_path: Optional[str] = None) -> bool:
        """Save a fine-tuned model."""
        try:
            if save_path is None:
                save_path = self.config.deep_learning.model_storage["base_path"]
            
            os.makedirs(save_path, exist_ok=True)
            
            if self.framework == "tensorflow":
                model_path = os.path.join(save_path, f"{model_name}_fine_tuned.h5")
                model["model"].save(model_path)
            else:
                model_path = os.path.join(save_path, f"{model_name}_fine_tuned.pth")
                torch.save(model["model"].state_dict(), model_path)
            
            # Save metadata
            metadata = {
                "framework": self.framework,
                "model_name": model_name,
                "base_model": model["base_model_info"]["type"],
                "created_at": datetime.now().isoformat(),
                "architecture": "transfer_learning"
            }
            
            metadata_path = os.path.join(save_path, f"{model_name}_fine_tuned_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Fine-tuned model saved to {model_path}")
            return True
            
        except Exception as e:
            error_handler.handle_error(f"Error saving fine-tuned model: {str(e)}", e)
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a fine-tuned model."""
        try:
            if model_name in self.fine_tuned_models:
                model_info = self.fine_tuned_models[model_name]
                return {
                    "model_name": model_name,
                    "base_model": model_info["base_model_info"]["type"],
                    "framework": self.framework,
                    "has_history": "history" in model_info,
                    "created_at": datetime.now().isoformat()
                }
            else:
                logger.warning(f"Fine-tuned model {model_name} not found")
                return {}
                
        except Exception as e:
            error_handler.handle_error(f"Error getting model info: {str(e)}", e)
            return {}
