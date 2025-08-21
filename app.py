import streamlit as st
from langchain.document_loaders import PyPDFLoader, DocxLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'

# Streamlit UI
st.title("Research Paper Summarizer")

# Upload document
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if uploaded_file:
    # Load the document based on file type
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = DocxLoader(uploaded_file)
    elif uploaded_file.type == "text/plain":
        loader = TextLoader(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    # Load and split the document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Summarization
    if st.button("Summarize Document"):
        llm = OpenAI()
        summary = llm("Summarize the following text:\n\n" + "\n".join([chunk.page_content for chunk in chunks]))
        st.subheader("Summary")
        st.write(summary)

    # Citation extraction
    if st.button("Extract Citations"):
        citation_extraction_prompt = "Extract all citations from the following text:\n\n" + "\n".join([chunk.page_content for chunk in chunks])
        citations = llm(citation_extraction_prompt)
        st.subheader("Citations")
        st.write(citations)

    # Search functionality
    query = st.text_input("Ask a question about the document:")
    if query:
        relevant_chunks = vector_store.similarity_search(query, k=3)
        response = llm("Answer the following question based on the chunks:\n" + query + "\n" + "\n".join([chunk.page_content for chunk in relevant_chunks]))
        st.subheader("Answer to Your Question")
        st.write(response)