from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_transformed, y_train)

    joblib.dump(model, 'models/model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
