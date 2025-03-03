import streamlit as st
import re
import pandas as pd
import io
import base64
from PIL import Image
import tempfile
import os
import sys
import cv2
import numpy as np

# Import OCR libraries
import pytesseract
from pdf2image import convert_from_bytes

#Import libraries for LLM
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from huggingface_hub import snapshot_download
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Loan Amount Extractor",
    page_icon="💰",
    layout="wide"
)

hf_token = st.secrets["READ_HUGGINGFACE"]

# App title and description
st.title("Loan Amount Extractor")
st.markdown("""
This app uses OCR and a fine-tuned language model to extract loan amounts from deeds of trust.
Upload a document (text, PDF, or image) or paste text to extract the loan amount.
""")

# Check if Tesseract is installed
def is_tesseract_installed():
    try:
        # Try to get tesseract version
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False

# Function to load your fine-tuned model
@st.cache_resource  # Cache the model to avoid reloading it every time
def load_model():
    model_dir = snapshot_download(repo_id="MaxMason42/Loan-Extraction",
                                  use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    
    st.success("Model loaded successfully!")
    return model, tokenizer

def scan_text(page):
    # Convert the page to grayscale
    gray = cv2.cvtColor(np.array(page), cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply thresholding
    _, binary_image = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8) 
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)
    
    # Convert image to PIL format
    pil_image = Image.fromarray(denoised)
    
    # Extract text using pytesseract
    custom_config = r'--oem 3 --psm 4'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return text


def chunk_text(text, tokenizer):
    def token_length_function(text):
        return len(tokenizer.encode(text))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=token_length_function,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    text_chunks = text_splitter.split_text(text)
    return text_chunks


# Function to extract loan amount using your model
def extract_loan_amount(text, model, tokenizer):
    def extract_price(text):
        input_text = text
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        device = model.device  # Get the model's device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(**inputs, max_length=20)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    text_chunks = chunk_text(text, tokenizer)
    results = []

    for chunk in text_chunks:
        price = extract_price(chunk)
        if price != "None":
            results.append({
                'price': price,
                'text': chunk
            })
    
    # Check if we found any results
    found = len(results) > 0
    
    # Return just the results array with chunks and prices
    return {
        "found": found,
        "results": results
    }

# Function to highlight text
def highlight_text(text, amount):
    """Highlight the loan amount in the text"""
    highlighted_text = text
    
    if amount:
        # Escape special regex characters in the amount
        escaped_amount = re.escape(amount)
        highlighted_text = re.sub(
            f'({escaped_amount})',
            r'<span style="background-color: #FFFF00; font-weight: bold;">\1</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    
    return highlighted_text

# Function to process uploaded file
def process_file(file):
    """Extract text from uploaded file using appropriate method"""
    text = ""
    
    # Check file type and read accordingly
    if file.name.endswith('.txt'):
        # Process text file
        text = file.getvalue().decode('utf-8')
    
    elif file.name.endswith('.pdf'):
        if not is_tesseract_installed():
            st.error("Tesseract OCR is not installed or not in PATH. Cannot process PDF files.")
            return ""
        
        try:
            st.info("Converting PDF to images...")
            # Convert PDF to images
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_bytes(file.getvalue())
                
                st.info(f"Performing OCR on {len(images)} pages...")
                # Process each page with OCR
                full_text = []
                for i, img in enumerate(images):
                    page_text = scan_text(img)
                    full_text.append(page_text)
                
                text = "\n".join(full_text)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return ""
    
    elif file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        if not is_tesseract_installed():
            st.error("Tesseract OCR is not installed or not in PATH. Cannot process image files.")
            return ""
        
        try:
            # Read image file
            image = Image.open(file)
            
            # Display image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            st.info("Performing OCR on image...")
            # Perform OCR
            text = scan_text(image)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return ""
    
    else:
        st.error("Unsupported file type. Please upload a TXT, PDF, or image file.")
    
    return text

# Check Tesseract installation at startup
tesseract_installed = is_tesseract_installed()
if not tesseract_installed:
    st.warning("""
    Tesseract OCR is not installed or not found in your PATH. 
    The app will only process text files, not PDFs or images.
    
    To install Tesseract:
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Mac: brew install tesseract
    - Linux: apt-get install tesseract-ocr
    
    After installation, make sure it's in your PATH or set pytesseract.pytesseract.tesseract_cmd to the full path.
    """)

# Load model when app starts
with st.spinner("Loading model..."):
    model, tokenizer = load_model()

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Upload Document", "Paste Text", "Try Examples"])

# Tab 1: Upload Document
with tab1:
    file_types = ["txt"]
    if tesseract_installed:
        file_types.extend(["pdf", "png", "jpg", "jpeg"])
    
    uploaded_file = st.file_uploader("Upload a deed of trust document", type=file_types)
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            text = process_file(uploaded_file)
        
        if text:
            st.subheader("Extracted Text")
            with st.expander("Show extracted text"):
                st.text(text)
            
            if st.button("Extract Loan Amount", key="extract_file"):
                with st.spinner("Extracting loan amount..."):
                    result = extract_loan_amount(text, model, tokenizer)
                    
                    if result["found"]:
                        st.success("Extraction complete!")
                        
                        # Display each found chunk with the extracted amount
                        for idx, r in enumerate(result["results"]):
                            st.subheader(f"Detected Amount: {r['price']}")
                            st.text_area(f"Text Chunk {idx+1}", r['text'], height=150)
                            
                            # Also show highlighted version
                            st.markdown("**Highlighted:**")
                            highlighted = highlight_text(r['text'], r['price'])
                            st.markdown(highlighted, unsafe_allow_html=True)
                            st.markdown("---")
                    else:
                        st.error("No loan amount could be detected in this document.")

# Tab 2: Paste Text
with tab2:
    text_input = st.text_area("Paste deed of trust text here", height=300)
    
    if text_input:
        if st.button("Extract Loan Amount", key="extract_text"):
            with st.spinner("Extracting loan amount..."):
                result = extract_loan_amount(text_input, model, tokenizer)
                
                if result["found"]:
                    st.success("Extraction complete!")
                    
                    # Display each found chunk with the extracted amount
                    for idx, r in enumerate(result["results"]):
                        st.subheader(f"Detected Amount: {r['price']}")
                        st.text_area(f"Text Chunk {idx+1}", r['text'], height=150)
                        
                        # Also show highlighted version
                        st.markdown("**Highlighted:**")
                        highlighted = highlight_text(r['text'], r['price'])
                        st.markdown(highlighted, unsafe_allow_html=True)
                        st.markdown("---")
                else:
                    st.error("No loan amount could be detected in this document.")

# Tab 3: Example Documents
with tab3:
    example_select = st.selectbox(
        "Select an example document",
        ["Example 1: $250,000 Loan", "Example 2: $375,000 Loan"]
    )
    
    example_texts = {
        "Example 1: $250,000 Loan": """DEED OF TRUST

THIS DEED OF TRUST is made this 15th day of June, 2023, between John Smith and Jane Smith, husband and wife (the "Borrower"), and First National Bank (the "Lender").

WHEREAS, Borrower is indebted to Lender in the principal sum of TWO HUNDRED FIFTY THOUSAND AND 00/100 DOLLARS ($250,000.00), which indebtedness is evidenced by Borrower's note dated June 15, 2023.""",
        
        "Example 2: $375,000 Loan": """DEED OF TRUST

THIS DEED OF TRUST ("Security Instrument") is made on April 10, 2023. The grantor is Robert Johnson and Mary Johnson ("Borrower"). The trustee is Heritage Trust Company ("Trustee"). The beneficiary is Mortgage Financial Inc., which is organized and existing under the laws of California and whose address is 123 Finance St., Los Angeles, CA 90001 ("Lender").

Borrower owes Lender the principal sum of Three Hundred Seventy-Five Thousand and 00/100 Dollars (U.S. $375,000.00). This debt is evidenced by Borrower's note dated the same date as this Security Instrument ("Note"), which provides for monthly payments."""
    }
    
    st.text_area("Example Document", example_texts[example_select], height=300)
    
    if st.button("Extract Loan Amount", key="extract_example"):
        with st.spinner("Extracting loan amount..."):
            result = extract_loan_amount(example_texts[example_select], model, tokenizer)
            
            if result["found"]:
                st.success("Extraction complete!")
                
                # Display each found chunk with the extracted amount
                for idx, r in enumerate(result["results"]):
                    st.subheader(f"Detected Amount: {r['price']}")
                    st.text_area(f"Text Chunk {idx+1}", r['text'], height=150)
                    
                    # Also show highlighted version
                    st.markdown("**Highlighted:**")
                    highlighted = highlight_text(r['text'], r['price'])
                    st.markdown(highlighted, unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.error("No loan amount could be detected in this document.")

# Add information about the model
st.sidebar.title("About")
st.sidebar.info("""
## Loan Amount Extractor

This application demonstrates a fine-tuned language model that extracts loan amounts from deeds of trust documents.

### How it works:
1. Upload a document (text, PDF, image) or paste text
2. OCR extracts text from non-text documents
3. The model analyzes the text to identify loan amounts
4. Results are displayed with highlights

### Model details:
- Fine-tuned on a dataset of deed of trust documents
- Integrated with Tesseract OCR for document processing
- Optimized for extracting loan amount information
""")

# Add portfolio information
st.sidebar.title("Portfolio Project")
st.sidebar.markdown("""
This is a demonstration of my work in natural language processing and machine learning.

Check out my other projects on my portfolio website!
""")