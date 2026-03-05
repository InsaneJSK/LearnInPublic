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
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("attention.pdf")
docs = loader.load()
docs

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
text_splitter.split_documents(docs)[:5]

# %%
documents=text_splitter.split_documents(docs)
documents

# %%
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

db=FAISS.from_documents(documents[:30],OllamaEmbeddings())

# %%
query="An attention function can be described as mapping a query "
result=db.similarity_search(query)
result[0].page_content

# %%
from langchain_community.llms import Ollama
## Load Ollama LAMA2 LLM model
llm=Ollama(model="llama2")

# %%
## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

# %%
## Create Stuff Docment Chain

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain=create_stuff_documents_chain(llm,prompt)

# %%
retriever=db.as_retriever()

# %%
from langchain.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)

# %%
response=retrieval_chain.invoke({"input":"Scaled Dot-Product Attention"})

# %%
print(response['answer'])
