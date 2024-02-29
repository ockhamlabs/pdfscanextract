import streamlit as st
import PyPDF2
import os
from tempfile import mkdtemp
import shutil
import zipfile
import json
from pdf2image import convert_from_path
import pytesseract
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import re

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)

def clean_text_for_pii_nltk(text):
    """
    Clean the text of PII using NLTK for NER to identify and redact names, cities,
    and organizations, and also remove any numbers.
    """
    # Simplified example; implement as needed
    return re.sub(r'\b\d+\b', '[NUMBER]', text)

#def perform_ocr(image_path):
    """
    #Perform OCR on an image using Tesseract.
    """
    #return pytesseract.image_to_string(image_path)

def extract_ngrams_and_sentences(text):
    """
    Extract n-grams and sentences from the cleaned text.
    """
    tokens = word_tokenize(text)
    # Extract trigrams
    trigrams = list(ngrams(tokens, 3)) if len(tokens) >= 3 else []
    # Extract sentences
    sentences = sent_tokenize(text)
    return {"trigrams": trigrams, "sentences": sentences}
###
import fitz  # PyMuPDF
import io
from PIL import Image
import json

def perform_ocr(image):
    # Use pytesseract to perform OCR on the image
    text = pytesseract.image_to_string(image)
    return text

def split_and_process_pdf(pdf_file, output_folder):
    doc = fitz.open(pdf_file)
    original_filename_prefix = os.path.splitext(os.path.basename(pdf_file.name))[0][:8]

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("ppm")
        
        # Convert the bytes to an image
        image = Image.open(io.BytesIO(img_bytes))
        
        # Perform OCR on the image
        text = perform_ocr(image)
        cleaned_text = clean_text_for_pii_nltk(text)  # Assuming this function is defined elsewhere
        
        # Save the processed text to a JSON file
        json_page_path = f"{output_folder}/json_outputs/{original_filename_prefix}_Page_{page_num+1:03d}.json"
        with open(json_page_path, 'w') as json_file:
            json.dump({"text": cleaned_text}, json_file)


#####

def zip_files(directory, zip_name):
    zip_filename = os.path.join(directory, f"{zip_name}.zip")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory)
                zipf.write(file_path, arcname=arcname)
    return zip_filename

def main():
    st.title('PDF Splitter and OCR App using Tesseract')

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        temp_dir = mkdtemp()
        os.makedirs(os.path.join(temp_dir, "pdf_pages"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "json_outputs"), exist_ok=True)

        # Split the PDF into individual pages and process each page
        split_and_process_pdf(uploaded_file, temp_dir)

        # Zip the split PDFs and JSON files
        pdf_zip_path = zip_files(os.path.join(temp_dir, "pdf_pages"), "split_pdfs")
        json_zip_path = zip_files(os.path.join(temp_dir, "json_outputs"), "json_outputs")

        # Offer ZIP files for download
        with open(pdf_zip_path, "rb") as f:
            st.download_button("Download Split PDFs as ZIP", f, "split_pdfs.zip", "application/zip")
        with open(json_zip_path, "rb") as f:
            st.download_button("Download JSON Outputs as ZIP", f, "json_outputs.zip", "application/zip")

        # Cleanup
        shutil.rmtree(temp_dir)  # Remove temporary directory and files after downloading

if __name__ == "__main__":
    main()
