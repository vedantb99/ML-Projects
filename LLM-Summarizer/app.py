import pdfplumber
from transformers import pipeline
import streamlit as st

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Initialize the summarization model (Hugging Face's BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to generate a summary
def generate_summary(text):
    # Truncate text if too long for the model
    max_input_length = 1024  # Hugging Face models typically have input limits
    if len(text) > max_input_length:
        text = text[:max_input_length]

    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Streamlit app
st.title("Intelligent Document Summarizer")

uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.write("Extracted Text:")
    st.text_area("Document Text", text, height=300)

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                summary = generate_summary(text)
                st.write("Summary:")
                st.success(summary)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
