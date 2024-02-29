import streamlit as st
import PyPDF2
import spacy
#nlp = spacy.load("en_core_web_sm")
#models = ["en_core_web_sm", "en_core_web_md"]
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
import pdf2image
import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image as PILImage
import json
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

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


from pdf2image import convert_from_path
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

def process_pdf_content(pdf_bytes):
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

def main():
    #st.title("PDF Processor for Text Extraction and PII Redaction")
    st.title("OCR Text Extraction")

    uploaded_files = st.file_uploader("Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            image = Image.open(io.BytesIO(bytes_data))
            text = extract_text_from_image(image)
            st.write("Extracted Text:", text)
    
    #uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    #if uploaded_files:
        #results = []
        #for uploaded_file in uploaded_files:
            #file_bytes = uploaded_file.read()
            #processed_data = process_pdf_content(file_bytes)
            #results.append({
                #"filename": uploaded_file.name,
                #"data": processed_data
            #})
        
        # Combine all results into a single JSON string for download
        json_results = json.dumps(results, indent=2)
        st.download_button(label="Download JSON Results",
                           data=json_results,
                           file_name="processed_data.json",
                           mime="application/json")

if __name__ == "__main__":
    main()
