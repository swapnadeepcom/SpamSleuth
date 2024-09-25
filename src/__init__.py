# __init__.py in src directory

from .data_preprocessing import load_data, split_data
from .model_training import train_model, save_model, load_model
from .model_inference import predict, evaluate_model
from .utils import clean_text, preprocess_text

# Defining a list of all publicly accessible methods for the package
__all__ = [
    "load_data", 
    "split_data", 
    "train_model", 
    "save_model", 
    "load_model", 
    "predict", 
    "evaluate_model",
    "clean_text", 
    "preprocess_text"
]

# Package version
__version__ = "1.0.0"

# Optional: Create an easier interface for running a full pipeline
def run_pipeline(data_path):
    """Run the full pipeline from data loading to model training and evaluation"""
    
    # Load and preprocess data
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data)

    # Train model
    model, vectorizer = train_model(X_train, y_train)

    # Evaluate model
    evaluation = evaluate_model(model, vectorizer, X_test, y_test)
    
    # Save the trained model and vectorizer
    save_model(model, vectorizer)
    
    return evaluation

