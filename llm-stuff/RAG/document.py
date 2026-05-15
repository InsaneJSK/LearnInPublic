# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from langchain_core.documents import Document

# %%
doc = Document(
    page_content="This is the content of the document.",
    metadata={"source": "example.pdf", "author": "John Doe"}
)
doc

# %%
import os
os.makedirs("data", exist_ok=True)

# %%
sample_texts = {
    "doc1.txt": "This is the content of document 1.",
    "doc2.txt": "This is the content of document 2.",
}
for filename, content in sample_texts.items():
    with open(f"data/{filename}", "w") as f:
        f.write(content)

# %%
from langchain_community.document_loaders import TextLoader
loader = TextLoader("data/doc1.txt")
documents = loader.load()

# %%
documents

# %%
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader, show_progress=True)
documents = loader.load()

# %%
documents

# %%
