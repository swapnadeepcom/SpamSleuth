# tests/test_model.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/model_inference.py')))

from model_inference import predict  # Import the function to be tested

def test_predict():
    """Test the predict function for spam classification"""
    spam_example = "Congratulations, you have won a free prize!"
    non_spam_example = "Can we reschedule our meeting to next week?"

    # Test spam prediction
    result_spam = predict(spam_example)
    assert result_spam == [1]  # Assuming 1 means 'spam'

    # Test non-spam prediction
    result_non_spam = predict(non_spam_example)
    assert result_non_spam == [0]  # Assuming 0 means 'not spam'
