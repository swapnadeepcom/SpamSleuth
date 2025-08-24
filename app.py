import streamlit as st
import pickle
import warnings

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the pipeline (vectorizer + model together)
try:
    pipeline = pickle.load(open("spam_classifier.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Title
st.title("📩 SMS Spam Classifier")
st.write("This app uses **Machine Learning (TF-IDF + Naive Bayes)** to classify messages as **Spam** or **Ham (Not Spam)**.")

# Input text box
input_sms = st.text_area("✉️ Enter your message here:")

# Predict button
if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first")
    else:
        # Predict using pipeline
        result = pipeline.predict([input_sms])[0]
        prob = pipeline.predict_proba([input_sms])[0]  # probability scores

        # Display result
        if result == 1:
            st.error("🚨 Spam Message")
            st.write(f"🔢 Confidence: {prob[1]*100:.2f}% Spam")
        else:
            st.success("✅ Not Spam (Ham)")
            st.write(f"🔢 Confidence: {prob[0]*100:.2f}% Ham")
