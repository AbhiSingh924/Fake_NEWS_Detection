import streamlit as st
import joblib
import os

# Set Streamlit page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Page title
st.title("🕵️‍♂️ Fake News Detection App")
st.markdown("Enter a news headline or article to check if it's **Real** or **Fake**, and verify it using Google.")

# Load the model
model_path = "fake_news_model.pkl"
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
else:
    st.error("❌ Model file not found. Please place `fake_news_model.pkl` in the same folder as this app.")

# Input field
user_input = st.text_area("📝 Enter News Text", height=200)

# Prediction
if st.button("Detect"):
    if not model:
        st.error("Model not loaded.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter some news content.")
    else:
        try:
            prediction = model.predict([user_input])[0]
            if prediction == 1:
                st.success("✅ This news appears to be **REAL**.")
            else:
                st.error("❌ This news appears to be **FAKE**.")

            # Add Google search link for verification
            search_query = f"https://www.google.com/search?q={user_input}+fact+check"
            st.markdown(
                f"[🔍 Verify this news on Google Fact Check]({search_query})",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
