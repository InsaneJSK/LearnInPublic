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
import nltk

# %%
from nltk.corpus import brown
print(brown.categories())

# %%
data = brown.sents(categories='adventure')

# %%
for i in data:
    print(" ".join(i))

# %%
document = """It was a pleasant day. The weather was cool and there were light showers. I went to the market to buy some fruits"""
sentence = "Send all the 50 documents related to chapters 1,2,3 at prateek@cb.com"

# %%
from nltk.tokenize import sent_tokenize, word_tokenize

# %%
import nltk
import os

# Auto-append path if not already there
punkt_path = "C:/Users/JSK/AppData/Roaming/nltk_data"
if punkt_path not in nltk.data.path:
    nltk.data.path.append(punkt_path)

# %%
sents = sent_tokenize(document)
print(sents)
print(len(sents))

# %%
sents[0]

# %%
words = word_tokenize(sentence)

# %%
words

# %%
#StopWord Removal
from nltk.corpus import stopwords
sw = stopwords.words('english')

# %%
sw


# %%
def remove_stopwords(text, stopwords):
    useful_words = [w for w in text if w not in stopwords]
    return useful_words


# %%
text = "i am not bothered about her very much".split()
useful_text = remove_stopwords(text, sw)

# %%
useful_text

# %%
from nltk.tokenize import RegexpTokenizer

# %%
tokenizer = RegexpTokenizer('[a-zA-Z@.]+')
useful_text = tokenizer.tokenize(sentence)

# %%
useful_text

# %%
from nltk.stem.snowball import SnowballStemmer
ss = SnowballStemmer("english")

# %%
ss.stem("jumping")

# %%
corpus = [
    "Indian cricket team will wins World Cup, says Capt. Virat Kohlt, world cup will be held at Sri Lanka",
    "We will win next Lok Sabha Elections, says confident Indian PM",
    "The nobel Laurate won the hearts of the people.",
    "The novie Raazi is an exciting Indian Spy thriller based upon a rest story."
]

# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
cv = CountVectorizer()

# %%
vectorized_corpus = cv.fit_transform(corpus)

# %%
vectorized_corpus.toarray()[0]

# %%
print(cv.vocabulary_)

# %%
len(cv.vocabulary_.keys())

# %%
# Reverse Mapping!
numbers = vectorized_corpus[2]
numbers

# %%
s = cv.inverse_transform(numbers)
print(s)


# %%
#Vectorization with Stopword Removal
def myTokenizer(document):
    words = tokenizer.tokenize(document.lower())
    # Remove Stopwords
    words = remove_stopwords(words,sw)
    return words


# %%
cv = CountVectorizer(tokenizer=myTokenizer)

# %%
vectorized_corpus = cv.fit_transform(corpus).toarray()

# %%
print(vectorized_corpus)

# %%
print(len(vectorized_corpus[0]))

# %%
cv.inverse_transform(vectorized_corpus)

# %%
# For Test Data
test_corpus = [
        'Indian cricket rock !',        
]

# %%
cv.transform(test_corpus).toarray()

# %% [markdown]
# ### More ways to Create Features
# - Unigram - every word as a feature
# - Bigrams
# - Trigrams
# - n-grams
# - TF-IDF Normalisation

# %%
sent_1  = ["this is good movie"]
sent_2 = ["this is good movie but actor is not present"]
sent_3 = ["this is not good movie"]

# %%
cv = CountVectorizer(ngram_range=(1,3))

# %%
docs = [sent_1[0],sent_2[0]]
cv.fit_transform(docs).toarray()

# %%
cv.vocabulary_

# %% [markdown]
# ### Tf-idf Normalisation
# - Avoid features that occur very often, becauase they contain less information
# - Information decreases as the number of occurences increases across different type of documents
# - So we define another term - term-document-frequency which associates a weight with every term

# %%
sent_1  = "this is good movie"
sent_2 = "this was good movie"
sent_3 = "this is not good movie"

corpus = [sent_1,sent_2,sent_3]

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
tfidf = TfidfVectorizer()

# %%
vc = tfidf.fit_transform(corpus).toarray()

# %%
print(vc)

# %%
tfidf.vocabulary_

# %%
from nltk import word_tokenize
sent = "Hey! Welcome to Coding Blocks ?."
words = set(word_tokenize(sent))

# %%
words
