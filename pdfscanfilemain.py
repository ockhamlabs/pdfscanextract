import streamlit as st
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from pdf2image import convert_from_bytes
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import json
import io
import requests
import re

# Ensure NLTK data is available
import nltk
nltk.download('punkt')

def clean_text_for_pii_nltk(text):
    """
    Clean the text of PII using NLTK for NER to identify and redact names, cities,
    and organizations, and also remove any numbers.
    """
    # Tokenize and tag text
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    # Perform NER
    entities = nltk.ne_chunk(tags)
    
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

# Function to perform OCR using Azure Computer Vision API
def perform_ocr(image_bytes, api_key, region, endpoint):
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(api_key))
    raw_response = computervision_client.read_in_stream(image_bytes, raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    operation_location_remote = raw_response.headers["Operation-Location"]
    
    # Extracting of text, wait 2 seconds, then get the result
    import time
    time.sleep(2)
    result = computervision_client.get_read_result(operation_location_remote)
    if result.status == OperationStatusCodes.succeeded:
        text = ''
        for text_result in result.analyze_result.read_results:
            for line in text_result.lines:
                text += line.text + '\n'
        return text
    else:
        return "No text detected."

# Function to extract n-grams and sentences
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

def process_pdf(pdf_bytes, pdf_name, api_key, region, endpoint):
    try:
        images = convert_from_bytes(pdf_bytes)
        all_pages_data = []
        for page_num, image in enumerate(images, start=1):
            text = perform_ocr(io.BytesIO(image.tobytes()), api_key, region, endpoint)
            # Assuming clean_text_for_pii_nltk and extract_ngrams_and_sentences functions are defined elsewhere
            cleaned_text = clean_text_for_pii_nltk(text)
            trigrams, sentences = extract_ngrams_and_sentences(cleaned_text)
            page_data = {
                "page_number": page_num,
                "text": cleaned_text,
                "trigrams": trigrams,
                "sentences": sentences
            }
            all_pages_data.append(page_data)
        return all_pages_data
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

def main():
    st.title("PDF Processor for Text Extraction with OCR")
    
    # Access Azure Vision API secrets from Streamlit secrets manager
    subscription_key = st.secrets["azure_subscription_key"]
    region = st.secrets["azure_region"]
    endpoint = st.secrets["azure_endpoint"]

    # File uploader for PDF files
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read PDF file as bytes
            pdf_bytes = uploaded_file.read()
            
            # Process the PDF and perform OCR
            extracted_data = process_pdf(pdf_bytes, uploaded_file.name, subscription_key, region, endpoint)
            
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
