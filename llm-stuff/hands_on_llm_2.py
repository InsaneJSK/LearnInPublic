# !pip install transformers>=4.40.1 accelerate>=0.27.2

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

generation_output = model.generate(input_ids=input_ids, max_new_tokens=20)

print(tokenizer.decode(generation_output[0]))

ctr = 1
for id in input_ids[0]:
  print(tokenizer.decode(id), end = "\t")
  if ctr%5 == 0:
    print()
  ctr += 1

print(generation_output)

ctr = 1
for id in generation_output[0]:
  print(tokenizer.decode(id), end = "\t")
  if ctr%5 == 0:
    print()
  ctr += 1

colors_list = [
'102;194;165', '252;141;98', '141;160;203',
'231;138;195', '166;216;84', '255;217;47'
]

text = """
English and CAPITALIZATION
🎵鸟
show_tokens False None elif == >= else: two tabs:" " Three tabs:
" "
12.0*50=600
"""

def show_tokens(sentence, tokenizer_name):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  token_ids = tokenizer(sentence).input_ids
  for idx, t in enumerate(token_ids):
    print(f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' + tokenizer.decode(t) + '\x1b[0m', end=' ')

show_tokens(text, "google-bert/bert-base-uncased")

show_tokens(text, "google-bert/bert-base-cased")

show_tokens(text, "openai-community/gpt2")

show_tokens(text, "google/flan-t5-xxl")

show_tokens(text, "Xenova/gpt-4")

show_tokens(text, "facebook/galactica-1.3b")

show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")

from transformers import AutoModel, AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# Tokenize the sentence
tokens = tokenizer('Hello world', return_tensors='pt')

# Process the tokens
output = model(**tokens)[0]

output.shape

for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))

output

from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Convert text to text embeddings
vector = model.encode("Best movie ever!")

vector.shape

# !pip install gensim

import gensim.downloader as api

# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
# Other options include "word2vec-google-news-300"
# More options at https://github.com/RaRe-Technologies/gensim-data
model = api.load("glove-wiki-gigaword-50")

model.most_similar([model['king']], topn=11)

import pandas as pd
from urllib import request

# Get the playlist dataset file
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')

# Parse the playlist dataset file. Skip the first two lines as
# they only contain metadata
lines = data.read().decode("utf-8").split('\n')[2:]

# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# Load song metadata
songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns = ['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')

print( 'Playlist #1:\n ', playlists[0], '\n')
print( 'Playlist #2:\n ', playlists[1])

from gensim.models import Word2Vec

# Train our Word2Vec model
model = Word2Vec(
    playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4
)

song_id = 2172

# Ask the model for songs similar to song #2172
model.wv.most_similar(positive=str(song_id))

print(songs_df.iloc[2172])

import numpy as np

def print_recommendations(song_id):
    similar_songs = np.array(
        model.wv.most_similar(positive=str(song_id),topn=5)
    )[:,0]
    return  songs_df.iloc[similar_songs]

# Extract recommendations
print_recommendations(2172)

print_recommendations(2172)

print_recommendations(842)

