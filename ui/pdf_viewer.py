"""
PDF viewer components for the Streamlit application.
"""

import streamlit as st
import pdfplumber

from processing.vector_db import create_simple_vector_db, delete_vector_db
from processing.document_processor import extract_all_pages_as_images


def render_pdf_uploader():
    """Render PDF file uploader and return the uploaded file"""
    return st.file_uploader(
        "Upload a PDF file ↓", 
        type = "pdf", 
        accept_multiple_files = False,
        key = "pdf_uploader"
    )


def handle_pdf_upload(file_upload):
    """Handle PDF file upload and processing"""
    if file_upload and st.session_state["vector_db"] is None:
        with st.spinner("Processing uploaded PDF..."):
            try:
                st.session_state["vector_db"] = create_simple_vector_db(file_upload)
                st.session_state["file_upload"] = file_upload
                # Extract PDF pages as images
                st.session_state["pdf_pages"] = extract_all_pages_as_images(file_upload)
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")


def render_pdf_viewer():
    """Render PDF viewer with zoom controls"""
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        zoom_level = st.slider(
            "Zoom Level", 
            min_value = 100, 
            max_value = 1000, 
            value = 700, 
            step = 50,
            key = "zoom_slider"
        )

        with st.container(height = 410, border = True):
            for page_image in st.session_state["pdf_pages"]:
                st.image(page_image, width = zoom_level)


def render_delete_button():
    """Render delete collection button"""
    delete_collection = st.button(
        "⚠️ Delete collection", 
        type = "secondary",
        key = "delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])