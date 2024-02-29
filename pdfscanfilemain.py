import streamlit as st
import PyPDF2
import spacy
import pytesseract
from PIL import Image as PILImage
import io
import json
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import tempfile
import os
import shutil
from pdf2image import convert_from_bytes  # Add this import statement
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

# OCR.space API endpoint
api_url = 'https://api.ocr.space/parse/image'

# API key (replace 'your-api-key' with your actual API key)
api_key = 'K84592797788957'

# Ensure NLTK data is available
import nltk
nltk.download('punkt')

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
import re

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def clean_text_for_pii_nltk(text):
    """
    Clean the text of PII using NLTK for NER to identify and redact names, cities,
    and organizations, and also remove any numbers.
    """
    # Tokenize and tag text
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)

    # Perform NER
    entities = ne_chunk(tags)
    
    # Helper function to traverse the named entities tree
    def traverse_tree(tree):
        entity_names = []
        if hasattr(tree, 'label') and tree.label:
            if tree.label() == 'PERSON' or tree.label() == 'GPE' or tree.label() == 'ORGANIZATION':
                for child in tree:
                    entity_names.append(' '.join([token for token, pos in child.leaves()]))
        return entity_names

    # Redact named entities
    for subtree in entities.subtrees(filter=lambda t: t.label() in ['PERSON', 'GPE', 'ORGANIZATION']):
        for entity in traverse_tree(subtree):
            text = text.replace(entity, "[REDACTED]")

    # Redact numbers
    text = re.sub(r'\b\d+\b', '[NUMBER]', text)
    return text

import streamlit as st
import requests
import io
import tempfile
import json
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path

# Function to perform OCR using OCR.space API
def perform_ocr(pdf_file):
    # Prepare data for POST request
    data = {
        'apikey': api_key,
        'language': 'eng',  # Language code for English
        'isOverlayRequired': False,
    }
    
    # Send POST request to OCR.space API with PDF file
    files = {'file': pdf_file}
    response = requests.post(api_url, data=data, files=files)
    
    # Check if request was successful
    if response.status_code == 200:
        # Parse JSON response
        result = response.json()
        
        # Extract and return text
        if 'ParsedResults' in result and result['ParsedResults']:
            return result['ParsedResults'][0]['ParsedText']
        else:
            return "No text detected."
    else:
        return "Error performing OCR."


def extract_ngrams_and_sentences(text):
    tokens = word_tokenize(text)
    trigrams_list = list(ngrams(tokens, 3))
    sentences = sent_tokenize(text)
    valid_sentences = [sentence for sentence in sentences if len(word_tokenize(sentence)) >= 3]
    three_four_word_sentences = {
        "three_words": [" ".join(word_tokenize(sentence)[:3]) for sentence in valid_sentences],
        "four_words": [" ".join(word_tokenize(sentence)[:4]) for sentence in valid_sentences if len(word_tokenize(sentence)) >= 4]
    }
    return Counter(trigrams_list), three_four_word_sentences


#from pdf2image import convert_from_path
import tempfile

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

#def extract_text_from_images(images):
    #texts = []
    #for image in images:
        #text = pytesseract.image_to_string(image)
        #texts.append(text)
    #return texts

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def process_pdf(pdf_bytes, pdf_name):
    try:
        images = convert_from_bytes(pdf_bytes)
        all_pages_data = []
        for page_num, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image)
            # Assuming clean_text_for_pii_nltk and extract_ngrams_and_sentences functions are defined elsewhere
            cleaned_text = clean_text_for_pii_nltk(text)
            trigrams, sentences = extract_ngrams_and_sentences(cleaned_text)
            page_data = {
                "page_number": page_num,
                "trigrams": trigrams,
                "sentences": sentences
            }
            all_pages_data.append(page_data)
        return all_pages_data
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

def main():
    st.title("PDF Processor with Online OCR")

    # File uploader for PDF files
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Perform OCR on uploaded PDF file
        extracted_text = perform_ocr(uploaded_file)
 
        
        # Process the PDF and perform OCR
        #extracted_data = process_pdf(pdf_bytes, uploaded_file.name)
        
        # Create JSON output
        output_data = {
            "pdf_name": uploaded_file.name,
            "pages": extracted_data
        }
        
        # Write JSON output to file
        output_file_path = f"{uploaded_file.name}_output.json"
        with open(output_file_path, "w") as json_file:
            json.dump(output_data, json_file)
        
        # Show link to download JSON file
        st.markdown(f"Download JSON output: [Download {uploaded_file.name}_output.json](/{output_file_path})")


if __name__ == "__main__":
    main()
