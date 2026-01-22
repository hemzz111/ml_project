import streamlit as st
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- CSS Styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #0f111a;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 8em;
        border-radius: 10px;
        border: none;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Load CSV ---
st.title("📧 Spam Email Detector")
st.write("Type your email below and check if it is **Spam** or **Not Spam**.")

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]  # Keep only label and message columns
    df = df.rename(columns={'v1':'label', 'v2':'message'})
    df['label'] = df['label'].map({'ham':0, 'spam':1})
    return df

data = load_data()

# --- Train model ---
@st.cache_data
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    # Test accuracy
    X_test_vec = vectorizer.transform(X_test)
    acc = accuracy_score(y_test, model.predict(X_test_vec))
    return model, vectorizer, acc

model, vectorizer, accuracy = train_model(data)
st.write(f"Model trained successfully! Accuracy: **{accuracy*100:.2f}%**")

# --- Clean text function ---
def clean_text_input(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# --- User input ---
user_input = st.text_area("Enter your email message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter an email message!")
    else:
        user_input_clean = [clean_text_input(user_input)]
        user_vector = vectorizer.transform(user_input_clean)
        prediction = model.predict(user_vector)
        if prediction[0] == 1:
            st.error("🚨 This is Spam!")
        else:
            st.success("✅ This is Not Spam!")

# --- Optional: Test multiple messages ---
st.subheader("Test Multiple Messages")
test_messages = [
    "Congratulations! You have won a free prize",
    "Hi, are we meeting today?",
    "Get a loan approved instantly!"
]

test_vector = vectorizer.transform([clean_text_input(msg) for msg in test_messages])
preds = model.predict(test_vector)

for msg, pred in zip(test_messages, preds):
    label = "Spam 🚨" if pred == 1 else "Not Spam ✅"
    st.write(f"Message: {msg}")
    st.write(f"Prediction: {label}")
    st.write("---")
