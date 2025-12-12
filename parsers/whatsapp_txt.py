def read_txt_file(uploaded_file) -> str:
    """
    Read a Streamlit-uploaded TXT file safely.
    """
    data = uploaded_file.read()
    return data.decode("utf-8", errors="ignore")
