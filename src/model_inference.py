import joblib

def load_model():
    model = joblib.load('models/model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return model, vectorizer

def predict(text):
    model, vectorizer = load_model()
    text_transformed = vectorizer.transform([text])
    return model.predict(text_transformed)
