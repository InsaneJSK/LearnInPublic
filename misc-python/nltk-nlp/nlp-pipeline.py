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
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# %%
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words("english"))
ps = PorterStemmer()


# %%
def getStemmedReview(review):
    review = review.lower()
    review = review.replace("<br /><br />", " ")

    #Tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    cleaned_review = " ".join(stemmed_tokens)
    return cleaned_review


# %%
def getStemmedDocument(inputfile, outputfile):
    with open(inputfile, "r") as f:
        reviews = f.readlines()
    with open(outputfile, "w") as f:
        for review in reviews:
            cleaned_review = getStemmedReview(review)
            f.write(cleaned_review)
