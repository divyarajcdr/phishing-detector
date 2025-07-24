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

# --- Set Streamlit page configuration FIRST ---
st.set_page_config(page_title="Intelligent Phishing Detector", page_icon="🛡️", layout="wide")

# --- Pre-loading and Function Definitions ---

@st.cache_resource
def load_resources():
    """Load the saved model, vectorizer, and NLTK data."""
    model = None
    vectorizer = None
    
    # Define the expected file paths relative to the app.py script
    model_path = 'phishing_detector_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    # Check if files exist before trying to load
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Model file not found at {model_path}. Please ensure 'phishing_detector_model.pkl' is in the same directory as app.py.")
        return None, None # Return None if files are not found

    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Error: Vectorizer file not found at {vectorizer_path}. Please ensure 'tfidf_vectorizer.pkl' is in the same directory as app.py.")
        return None, None # Return None if files are not found

    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
    except Exception as e:
        st.error(f"❌ Error loading model or vectorizer: {e}. Please check file integrity.")
        return None, None # Return None on loading error

    # Download NLTK data (only if not already downloaded)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    return model, vectorizer

# Call the function to load everything when the app starts
loaded_model, loaded_vectorizer = load_resources()

# Initialize NLTK components for preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    Steps include:
    - Removing URLs
    - Removing HTML tags
    - Removing non-alphabetic characters
    - Converting text to lowercase
    - Tokenization
    - Lemmatization
    - Stop word removal
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

# --- Main Classification Function ---
def classify_content(text_content):
    """
    Preprocesses the given text content, vectorizes it, and uses the loaded model
    to predict if it's legitimate or phishing. Displays the result and confidence.
    """
    if not text_content.strip():
        st.warning("⚠️ Please provide some text to analyze.")
        return

    # Ensure model and vectorizer are loaded before proceeding
    if loaded_model is None or loaded_vectorizer is None:
        st.error("Model or vectorizer not loaded. Cannot perform classification.")
        return

    processed_text = preprocess_text(text_content)
    vectorized_text = loaded_vectorizer.transform([processed_text])

    # Manually add the 'urls' feature. 
    urls_feature = scipy.sparse.csr_matrix([[0]])

    combined_features = scipy.sparse.hstack([vectorized_text, urls_feature])

    prediction = loaded_model.predict(combined_features)[0]
    prediction_proba = loaded_model.predict_proba(combined_features)[0]

    st.write("---") 
    st.subheader("Analysis Result")
    if prediction == 1: 
        confidence = prediction_proba[1] 
        st.error(f"Result: PHISHING 🎣 (Confidence: {confidence:.2%})")
        st.info("This content exhibits characteristics commonly found in phishing attempts. Exercise extreme caution.")
    else: 
        confidence = prediction_proba[0] 
        st.success(f"Result: LEGITIMATE ✅ (Confidence: {confidence:.2%})")
        st.info("This content appears legitimate. However, always remain vigilant with suspicious emails.")

# --- Page Navigation and UI Setup ---

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_to_analyzer():
    st.session_state.page = 'analyzer'

def go_to_home():
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    st.title("🛡️ Welcome to the Intelligent Phishing Detector")
    st.markdown("---")
    
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
    if loaded_model is None or loaded_vectorizer is None:
        st.error("❌ Application cannot run: Model or vectorizer files are missing or could not be loaded. Please ensure 'phishing_detector_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as this app in your GitHub repository.")
        st.button("Go Back to Home", on_click=go_to_home)
    else:
        st.title("📧 Phishing Email Analyzer")
        st.button("← Back to Home", on_click=go_to_home)
        
        tab1, tab2 = st.tabs(["Paste Email Text", "Upload Image"])

        with tab1:
            st.header("Analyze Pasted Text")
            text_area = st.text_area("Paste the full email text here...", height=250, placeholder="e.g., 'Dear customer, your account has been suspended. Click here to verify.'")
            if st.button("Classify Text"):
                classify_content(text_area)

        with tab2:
            st.header("Analyze from an Image")
            uploader = st.file_uploader("Upload a screenshot of the email (.png, .jpg, .jpeg)", type=['png', 'jpg', 'jpeg'])
            if uploader is not None:
                try:
                    image = Image.open(uploader)
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    
                    extracted_text = pytesseract.image_to_string(image)
                    
                    st.write("### Extracted Text:")
                    st.text_area("Text found in image:", value=extracted_text, height=150, disabled=True)
                    
                    if st.button("Classify Image Text"): 
                        classify_content(extracted_text)
                except pytesseract.TesseractNotFoundError:
                    st.error("Tesseract is not installed or not in your PATH. Please install it to use image analysis. Refer to the Streamlit documentation or Tesseract's GitHub for installation instructions.")
                except Exception as e:
                    st.error(f"An error occurred during image processing: {e}. Please try another image.")

