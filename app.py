import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

# ---------------- NLTK FIX (IMPORTANT) ----------------
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

ps = PorterStemmer()

# ---------------- UI Styling ----------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00ADB5;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 20px;
}
.result-spam {
    color: #FF4B4B;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}
.result-ham {
    color: #00C897;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<div class="title">📩 SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a message is Spam or Not</div>', unsafe_allow_html=True)

# ---------------- Input Box ----------------
input_sms = st.text_area("✉️ Enter your message here:", height=150)

# ---------------- Preprocessing ----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# ---------------- Button ----------------
if st.button("🚀 Predict"):
    
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # ---------------- Result ----------------
        st.markdown("---")

        if result == 0:
            st.markdown('<div class="result-ham">✅ Not Spam</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-spam">🚨 Spam Message</div>', unsafe_allow_html=True)