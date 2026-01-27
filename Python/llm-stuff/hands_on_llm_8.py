# -*- coding: utf-8 -*-
"""Hands-on-llm-chap-8
"""

!pip install cohere faiss-cpu rank_bm25 --quiet

import cohere
import numpy as np
import pandas as pd
from tqdm import tqdm
api_key = 'YOUR_COHERE_API_KEY'

co = cohere.Client(api_key)

text = """
Interstellar is a 2014 epic science fiction film directed by Christopher Nolan, who co-wrote the screenplay with his brother Jonathan Nolan. It features an ensemble cast led by Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, and Michael Caine. Set in a dystopian future where Earth is suffering from catastrophic blight and famine, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.

The screenplay had its origins in a script that Jonathan had developed in 2007 and was originally set to be directed by Steven Spielberg. Theoretical physicist Kip Thorne was an executive producer and scientific consultant on the film, and wrote the tie-in book The Science of Interstellar. It was Lynda Obst's final film as producer before her death. Cinematographer Hoyte van Hoytema shot it on 35 mm film in the Panavision anamorphic format and IMAX 70 mm. Filming began in late 2013 and took place in Alberta, Klaustur, and Los Angeles. Interstellar uses extensive practical and miniature effects, and the company DNEG created additional visual effects.

Interstellar premiered at the TCL Chinese Theatre on October 26, 2014, and was released in theaters in the United States on November 5, and in the United Kingdom on November 7, with Paramount Pictures distributing in the United States and Warner Bros. Pictures distributing in international markets. In the United States, it was first released on film stock, expanding to venues using digital projectors. It was a commercial success, grossing $681 million worldwide during its initial theatrical run, and $773.8 million worldwide with subsequent releases, making it the 10th-highest-grossing film of 2014. The film received generally positive reviews from critics. Among its various accolades, Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects.
"""
texts = text.split('.')

texts = [t.strip(' \n') for t in texts]

response = co.embed(
    texts = texts,
    input_type = "search_document",
).embeddings

embeds = np.array(response)
print(embeds.shape)

import faiss
dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
print(index.is_trained)
index.add(np.float32(embeds))

def search(query, number_of_results=3):
  query_embed = co.embed(texts = [query], input_type = "search_query",).embeddings[0]
  distances, similar_item_ids = index.search(np.float32([query_embed]), number_of_results)
  texts_np = np.array(texts)
  results = pd.DataFrame(data={'texts':texts_np[similar_item_ids[0]], 'distances':distances[0]})
  print(f'Query: "{query}"\nNearest neighbours:')
  return results

query = "how precise was the science?"
results = search(query)
results

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS
import string
def bm25_tokenizer(text):
  tokenized_doc = []
  for token in text.lower().split():
    token = token.strip(string.punctuation)

    if len(token) > 0 and token not in stop_words:
      tokenized_doc.append(token)
  return tokenized_doc

tokenized_corpus = []
for passage in tqdm(texts):
  tokenized_corpus.append(bm25_tokenizer(passage))
bm25 = BM25Okapi(tokenized_corpus)

def keyword_search(query, top_k = 3, num_candidates = 15):
  print("Input question: ", query)
  bm25_scores = bm25.get_scores(bm25_tokenizer(query))
  top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
  bm25_hits = [{'corpus_id': idx, 'score':bm25_scores[idx]} for idx in top_n]
  bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
  print(f"Top-3 lexical search (BM25) hits")
  for hit in bm25_hits[0:top_k]:
    print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))

keyword_search(query = "How precise was the science")
