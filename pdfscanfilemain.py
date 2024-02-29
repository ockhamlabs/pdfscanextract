import streamlit as st
import PyPDF2
import spacy
import spacy_streamlit
#nlp = spacy.load("en_core_web_sm")
models = ["en_core_web_sm", "en_core_web_md"]
import pytesseract
from PIL import Image as PILImage
from wand.image import Image as WandImage
import io
import json
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import tempfile
import os
import shutil

# Ensure NLTK data is available
import nltk
nltk.download('punkt')

# Initialize spaCy for NER
nlp = spacy_streamlit.load_model("en_core_web_sm")

import spacy
import sys

def download_spacy_model(model_name="en_core_web_sm"):
    """
    Ensure spaCy model is downloaded.
    """
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Downloading the spaCy model {model_name}...")
        from spacy.cli import download
        download(model_name)
        # Ensure the downloaded model is loaded
        return spacy.load(model_name)

nlp = download_spacy_model()


def clean_text_for_pii(text):
    """
    Clean the text of PII using NER to identify and redact names, cities, and organizations,
    and also remove any numbers.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            text = text.replace(ent.text, "[REDACTED]")
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

def pdf_to_images(pdf_file, resolution=300):
    images = []
    with WandImage(file=pdf_file, resolution=resolution) as img:
        for page in img.sequence:
            with WandImage(page) as page_img:
                page_png = page_img.make_blob('png')
                images.append(PILImage.open(io.BytesIO(page_png)))
    return images

def extract_text_from_images(images):
    texts = []
    for image in images:
        text = pytesseract.image_to_string(image)
        texts.append(text)
    return texts

def process_pdf(pdf_file):
    images = pdf_to_images(pdf_file)
    all_pages_data = []
    for page_num, image in enumerate(images, start=1):
        text = extract_text_from_images([image])[0]  # Since it now returns a list of texts, take the first item
        cleaned_text = clean_text_for_pii(text)
        trigrams, sentences = extract_ngrams_and_sentences(cleaned_text)
        page_data = {
            "page_number": page_num,
            "trigrams": trigrams,
            "sentences": sentences
        }
        all_pages_data.append(page_data)
    return all_pages_data

def main():
    st.title("PDF Processor for Text Extraction and PII Redaction")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf_file:
            tmp_pdf_file.write(uploaded_file.read())
            pdf_file_path = tmp_pdf_file.name

        # Process the PDF
        with open(pdf_file_path, "rb") as pdf_file:
            processed_data = process_pdf(pdf_file)
        
        # Prepare and display the results
        st.write("Processed Data:", processed_data)

        # Optionally, convert processed data to JSON and allow download
        json_results = json.dumps(processed_data, default=str)  # Using default=str to handle non-serializable objects gracefully
        st.download_button(label="Download JSON Results",
                           data=json_results,
                           file_name="processed_data.json",
                           mime="application/json")

        # Clean up the temporary file
        os.remove(pdf_file_path)

if __name__ == "__main__":
    main()
