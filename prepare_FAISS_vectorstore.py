import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#function to load scraped documents
def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    

therapyDocuments = [
    Document(page_content=load_documents("data/therapy/meditations.txt"), metadata={"source": "Meditations"}),
    Document(page_content=load_documents("data/therapy/cleaveland.txt"), metadata={"source": "Cleaveland Clinic"}),
    Document(page_content=load_documents("data/therapy/who_psych_guidelines.txt"), metadata={"source": "WHO Guidelines"}),
    
]

resourceDocuments = [
    Document(page_content=load_documents("data/resources/global_helplines.txt"), metadata={"source": "Global Helplines"}),
    Document(page_content=load_documents("data/resources/india_helplines.txt"), metadata={"source": "Global Helplines"}),
    Document(page_content=load_documents("data/resources/free_resources_india.txt"), metadata={"source": "Free Resources India"}),
    Document(page_content=load_documents("data/resources/who_psych_guidelines.txt"), metadata={"source": "WHO Guidelines"}),
]

#Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_therapy_docs = splitter.split_documents(therapyDocuments)
split_resource_docs = splitter.split_documents(resourceDocuments)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#Build FAISS vectorstore
therapy_vectorstore = FAISS.from_documents(split_therapy_docs, embedding_model)
resource_vectorstore = FAISS.from_documents(split_resource_docs, embedding_model)
#Save vectorstore locally
therapy_vectorstore.save_local("faiss_therapy_index")
resource_vectorstore.save_local("faiss_resource_index")