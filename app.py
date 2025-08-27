import streamlit as st
import pickle
import warnings
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the pipeline
try:
    pipeline = pickle.load(open("spam_classifier.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ------------------ ULTRA MODERN THEME ------------------
theme_css = """
<style>
/* Animated Gradient Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Poppins', sans-serif;
    color: #111;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(12px);
    border-right: 2px solid rgba(255,255,255,0.3);
}

/* Title */
h1 {
    text-align: center;
    font-size: 3rem !important;
    font-weight: 900;
    background: linear-gradient(90deg, #00c6ff, #ff758c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.8rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.07);
    background: linear-gradient(135deg, #ff512f, #dd2476);
}

/* Text Area */
.stTextArea textarea {
    background: rgba(255,255,255,0.9);
    border: 2px solid rgba(255,255,255,0.4);
    border-radius: 14px;
    padding: 14px;
    font-size: 1.1rem;
    color: #111;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Result Card */
.result-box {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(14px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    text-align: center;
    margin-top: 20px;
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Confidence Text */
.confidence-text {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 10px;
    color: #222;
}

/* Wide section for charts using CSS grid */
.wide-charts {
    display: grid;
    grid-template-columns: 1fr 1.5fr 1.5fr;
    gap: 25px;
    padding-left: 5%;
    padding-right: 5%;
    margin-top: 25px;
}
.wide-charts > div {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.15);
}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# ------------------ APP CONTENT ------------------
st.markdown("<h1>📩 SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.write(
    "This app uses **Machine Learning (TF-IDF + Naive Bayes)** to classify messages as **Spam** or **Ham (Not Spam)**."
)

input_sms = st.text_area("✉️ Enter your message here:")

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first")
    else:
        result = pipeline.predict([input_sms])[0]
        prob = pipeline.predict_proba([input_sms])[0]

        labels = ["Ham (Not Spam)", "Spam"]
        values = [prob[0]*100, prob[1]*100]

        # Display result in a styled glassmorphism card
        if result == 1:
            st.markdown(
                f"""
                <div class="result-box">
                    <h2 style='color:#ff1744;'>🚨 Spam Message</h2>
                    <p class="confidence-text">🔢 Confidence: <b>{prob[1]*100:.2f}% Spam</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-box">
                    <h2 style='color:#2ecc71;'>✅ Not Spam (Ham)</h2>
                    <p class="confidence-text">🔢 Confidence: <b>{prob[0]*100:.2f}% Ham</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------- Full-width Section for Table + Charts ----------------
        st.markdown('<div class="wide-charts">', unsafe_allow_html=True)

        # Table
        st.markdown("<div>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='font-size:30px; font-weight:900; text-align:center;'>📋 Probabilities</h3>",
            unsafe_allow_html=True
        )
        df = pd.DataFrame({"Class": labels, "Probability (%)": [f"{v:.2f}" for v in values]})
        st.table(df)
        st.markdown("</div>", unsafe_allow_html=True)

        # Bar Chart
        st.markdown("<div>", unsafe_allow_html=True)
        fig = px.bar(
            x=labels,
            y=values,
            color=labels,
            text=[f"{v:.2f}%" for v in values],
            color_discrete_map={"Ham (Not Spam)": "#2ecc71", "Spam": "#e63946"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            title=dict(text="<b>Prediction Confidence</b>", font=dict(size=30, family="Poppins", color="#222")),
            yaxis=dict(title="Probability (%)", range=[0, 100]),
            xaxis=dict(title=f"Message: {input_sms}"),  # ✅ Changed here
            legend=dict(font=dict(size=20)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Pie Chart (transparent background)
        st.markdown("<div>", unsafe_allow_html=True)
        pie = px.pie(
            names=labels,
            values=values,
            color=labels,
            color_discrete_map={"Ham (Not Spam)": "#2ecc71", "Spam": "#e63946"},
            title="<b>Spam vs Ham</b>"
        )
        pie.update_layout(
            title=dict(font=dict(size=30, family="Poppins", color="#222")),
            legend=dict(font=dict(size=20)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
