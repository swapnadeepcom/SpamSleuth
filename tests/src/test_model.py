from src.model_inference import predict

def test_predict():
    result = predict("Free money now!!!")
    assert result == [1]  # Assuming 1 is for 'spam'
