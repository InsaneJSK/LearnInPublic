import nltk
import os

# Print NLTK version
print("NLTK version:", nltk.__version__)

# Print data paths
print("NLTK data paths:", nltk.data.path)

# Check if punkt exists
punkt_path = nltk.data.find('tokenizers/punkt')
print("Punkt path:", punkt_path)

# Test tokenization
from nltk.tokenize import sent_tokenize
document = "This is a test. This is another sentence."
sents = sent_tokenize(document)
print("Tokenized sentences:", sents)