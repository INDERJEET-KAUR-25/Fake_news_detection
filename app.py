import streamlit as st
import pickle
import re
import string

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def word_clean(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

def predict_news(text):
    text = word_clean(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return pred[0]

# 🎨 Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: black;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #7f8c8d;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 📰 Title
st.markdown('<div class="title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a news article is Real or Fake using Machine Learning</div>', unsafe_allow_html=True)

st.write("")

# 📥 Input Box
user_input = st.text_area("✍️ Enter News Content Below:", height=200)

# 🎯 Buttons
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict")

with col2:
    clear_btn = st.button("🧹 Clear")

# 🔮 Prediction
if predict_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text")
    else:
        result = predict_news(user_input)

        if result == 1:
            st.markdown('<div class="result-box" style="background-color:#d4edda; color:#155724;">🟢 This is Real News</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="background-color:#f8d7da; color:#721c24;">🔴 This is Fake News</div>', unsafe_allow_html=True)

# 🧹 Clear
if clear_btn:
    st.experimental_rerun()

# 📌 Footer
st.markdown("---")
st.markdown("👩‍💻 Developed by **Inderjeet Kaur** | Machine Learning Project")