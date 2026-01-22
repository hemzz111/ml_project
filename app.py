import streamlit as st
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------- CSS ----------------
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 8em;
}
.stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📧 Spam Email Detector")
st.write("Enter an email message to check whether it is **Spam** or **Not Spam**")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

data = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_data
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))

    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_model(data)
st.success(f"Model trained successfully! Accuracy: {accuracy*100:.2f}%")

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# ---------------- DEFAULT SPAM WORDS ----------------
spam_words = [
    "free", "win", "winner", "prize", "lottery", "money",
    "loan", "credit", "urgent", "click", "offer", "buy now","immediately"
]

def contains_spam_words(text):
    for word in spam_words:
        if word in text:
            return True
    return False

# ---------------- USER INPUT ----------------
user_input = st.text_area("✉️ Enter Email Message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        cleaned = clean_text(user_input)

        # Rule-based spam check
        if contains_spam_words(cleaned):
            st.error("🚨 This is Spam ")
        else:
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)

            if prediction[0] == 1:
                st.error("🚨 This is Spam")
            else:
                st.success("✅ This is Not Spam")

