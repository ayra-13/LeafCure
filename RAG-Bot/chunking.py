import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Initialize Cohere embeddings
embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

def process_pdfs(pdf_folder="pdfs"):
    """Loads PDFs, chunks text, and stores vector embeddings in ChromaDB."""
    
    documents = []
    
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            pages = loader.load()  # Load page-wise

            for page in pages:
                page.metadata["source"] = file  # Store file name
            
            documents.extend(pages)
    
    # Smart chunking with context overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk ~1000 characters
        chunk_overlap=200  # Overlap ensures context retention
    )
    
    chunks = text_splitter.split_documents(documents)  # Perform chunking
    
    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"✅ Processed {len(chunks)} chunks from {len(documents)} pages.")
    
    return vectorstore

# Run chunking
pdf_folder = "./data"
process_pdfs(pdf_folder)
