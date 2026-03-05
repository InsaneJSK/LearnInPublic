# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
## Data Ingestion
from langchain_community.document_loaders import TextLoader
loader=TextLoader("speech.txt")
text_documents=loader.load()
text_documents

# %%
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["groq_api_key"] = os.getenv("groq_api_key")

# %%
# web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4

## load,chunk and index the content of the html page

loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")

                     )))

text_documents=loader.load()

# %%
text_documents


# %%
## Pdf reader
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('attention.pdf')
docs=loader.load()

# %%
docs

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)
documents[:5]

# %%
documents

# %%
## Vector Embedding And Vector Store
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# %%
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(documents,embeddings)

# %%
db

# %%
query = "Who are the authors of attention is all you need research paper?"
retireved_results=db.similarity_search(query)
retireved_results

# %%
## FAISS Vector Database
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(documents[:15], embedding=embeddings)

# %%
