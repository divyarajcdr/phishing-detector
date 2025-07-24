import streamlit as st
import pickle
import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pytesseract
from PIL import Image
import io
import scipy.sparse

# --- Pre-loading and Function Definitions ---

# This function will run once and cache the resources
@st.cache_resource
def load_resources():
    """Load the saved model, vectorizer, and NLTK data."""
    # Load model and vectorizer
    try:
        with open('phishing_detector_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
    except FileNotFoundError:
        return None, None # Return None if files are not found

    # Download NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    return model, vectorizer

# Call the function to load everything
loaded_model, loaded_vectorizer = load_resources()

# Define the preprocessing function (must be the same as in training)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

# --- Main Classification Function ---
def classify_content(text_content):
    """Preprocesses and classifies the given text, showing results."""
    if not text_content.strip():
        st.warning("⚠️ Please provide some text to analyze.")
        return

    # 1. Preprocess and vectorize the text content
    processed_text = preprocess_text(text_content)
    vectorized_text = loaded_vectorizer.transform([processed_text])

    # 2. Manually add the 'urls' feature, assuming a value of 0 for new inputs.
    urls_feature = scipy.sparse.csr_matrix([[0]])

    # 3. Combine features to match the model's expected input shape
    combined_features = scipy.sparse.hstack([vectorized_text, urls_feature])

    # 4. Predict using the model
    prediction = loaded_model.predict(combined_features)[0]
    prediction_proba = loaded_model.predict_proba(combined_features)[0]

    # Display results
    st.write("---")
    st.subheader("Analysis Result")
    if prediction == 1:
        confidence = prediction_proba[1]
        st.error(f"Result: PHISHING 🎣 (Confidence: {confidence:.2%})")
    else:
        confidence = prediction_proba[0]
        st.success(f"Result: LEGITIMATE ✅ (Confidence: {confidence:.2%})")

# --- Page Navigation and UI ---

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Functions to change the page
def go_to_analyzer():
    st.session_state.page = 'analyzer'
def go_to_home():
    st.session_state.page = 'home'

st.set_page_config(page_title="Phishing Detector", page_icon="🛡️", layout="wide")

# Display content based on the current page
if st.session_state.page == 'home':
    st.title("🛡️ Welcome to the Intelligent Phishing Detector")
    st.markdown("---")
    # --- IMPORTANT: FILL IN YOUR DETAILS HERE ---
    st.header("Team Name: [Your Team Name]")
    st.subheader("Team Members:")
    st.markdown("""
    * [Member 1 Name]
    * [Member 2 Name]
    * [Member 3 Name]
    * [Add more members as needed]
    """)
    st.markdown("---")
    st.button("Proceed to Analyzer", on_click=go_to_analyzer, type="primary")

elif st.session_state.page == 'analyzer':
    # Check if model files are loaded correctly
    if loaded_model is None or loaded_vectorizer is None:
        st.error("❌ Error: Model files not found. Please ensure 'phishing_detector_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as this app.")
        st.button("Go Back to Home", on_click=go_to_home)
    else:
        st.title("📧 Phishing Email Analyzer")
        st.button("← Back to Home", on_click=go_to_home)
        
        tab1, tab2 = st.tabs(["Paste Email Text", "Upload Image"])

        with tab1:
            st.header("Analyze Pasted Text")
            text_area = st.text_area("Paste the full email text here...", height=250)
            if st.button("Classify Text"):
                classify_content(text_area)

        with tab2:
            st.header("Analyze from an Image")
            uploader = st.file_uploader("Upload a screenshot of the email (.png, .jpg, .jpeg)", type=['png', 'jpg', 'jpeg'])
            if uploader is not None:
                try:
                    # Use Tesseract OCR to extract text from the uploaded image
                    image = Image.open(uploader)
                    extracted_text = pytesseract.image_to_string(image)
                    st.write("### Extracted Text:")
                    st.text_area("Text found in image:", value=extracted_text, height=150, disabled=True)
                    classify_content(extracted_text)
                except Exception as e:
                    st.error(f"An error occurred during image processing: {e}")