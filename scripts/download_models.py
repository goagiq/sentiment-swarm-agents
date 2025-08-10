#!/usr/bin/env python3
"""
Download required models for sentiment analysis.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

# Model configurations
MODELS = {
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "audio": "facebook/wav2vec2-base-960h",
    "vision": "microsoft/DialoGPT-medium"
}


def download_model(model_name: str, model_type: str):
    """Download a specific model."""
    try:
        logger.info(f"Downloading {model_type} model: {model_name}")
        
        if model_type == "sentiment":
            # Download sentiment analysis pipeline
            _ = pipeline("sentiment-analysis", model=model_name)
        elif model_type == "audio":
            # Download audio model components
            _ = AutoTokenizer.from_pretrained(model_name)
            _ = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif model_type == "vision":
            # Download vision model components
            _ = AutoTokenizer.from_pretrained(model_name)
            _ = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        logger.success(f"Successfully downloaded {model_type} model")
        
    except Exception as e:
        logger.error(f"Failed to download {model_type} model: {e}")
        raise


def main():
    """Main function to download all models."""
    logger.info("Starting model download process...")
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Set environment variable for model cache
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    
    # Download each model
    for model_type, model_name in MODELS.items():
        try:
            download_model(model_name, model_type)
        except Exception as e:
            logger.error(f"Failed to download {model_type} model: {e}")
            continue
    
    logger.success("Model download process completed!")
    logger.info(f"Models stored in: {models_dir}")


if __name__ == "__main__":
    main()
