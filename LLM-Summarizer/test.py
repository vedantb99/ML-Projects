#https://www.youtube.com/watch?v=73k3d63qKqU&list=PLATp91001ydYSklYucxBjQDhZ_pdFYh2C&index=6
import streamlit as st
import os
from utils import *
def main():
    st.set_page_config(page_title="PDF Summarizer")
    st.title("PDF Summarizer")
    st.write("Summarize your pdf docs in a few seconds!!")
    st.divider()
    
    pdf = st.file_uploader("Kindly upload pdf: ", type="pdf")
    submit = st.button("Generate Summary")
    os.environ["OPENAI_API_KEY"] = "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"
    if submit:
        response = summarizer(pdf)
    st.subheader('Summary of file: ')
    st.write(response)
if __name__ == '__main__':
    main()