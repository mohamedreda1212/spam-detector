import streamlit as st
import pickle

# Load model and vectorizer
with open('spam_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# Page configuration
st.set_page_config(page_title="📨 Spam Classifier", page_icon="📩", layout="centered")

# Custom soft background and elegant styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f8fc;
        }
        h1, h3 {
            text-align: center;
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }
        .stTextArea label {
            font-weight: bold;
            color: #34495e;
            font-size: 16px;
        }
        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid;
            font-size: 18px;
            font-weight: bold;
        }
        .ham {
            background-color: #eafaf1;
            border-left-color: #2ecc71;
            color: #145a32;
        }
        .spam {
            background-color: #fdecea;
            border-left-color: #e74c3c;
            color: #922b21;
        }
    </style>
""", unsafe_allow_html=True)

# Title & description
st.title("📨 AI Spam Message Classifier")
st.markdown("💬 Enter your message below and our AI will classify it as **Spam** or **Ham (Safe)**.")

# User input
message = st.text_area("✉️ Message:")

# Predict button
if st.button("🚀 Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Vectorize input and predict
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.markdown(
                '<div class="result-box spam">🚫 This message is <strong>SPAM</strong>. Be cautious.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box ham">✅ This message is <strong>HAM</strong>. It looks safe.</div>',
                unsafe_allow_html=True
            )
