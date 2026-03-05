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
x = ["This was awesome an awesome movie",
     "Great movie! I liked it a lot",
     "Happy Ending! awesome acting by the hero",
     "loved it! truly great",
     "bad not upto the mark",
     "could have been better",
     "Surely a Disappointing movie"]

y = [1,1,1,1,0,0,0]

# %%
x_test = ["I was happy & happy and I loved the acting in the movie",
          "The movie I saw was bad"]

# %%
import clean_text as ct

# %%
x_clean = [ct.getCleanReview(i) for i in x] #List Comprehension
xt_clean = [ct.getCleanReview(i) for i in x_test]

# %%
print(x_clean)
print(xt_clean)

# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
cv = CountVectorizer(ngram_range=(1,2))

x_vec = cv.fit_transform(x_clean).toarray()
print(x_vec)
print(x_vec.shape)

# %%
print(cv.get_feature_names_out())

# %%
xt_vec = cv.transform(xt_clean).toarray()
print(xt_vec)
cv.get_feature_names_out()
print(xt_vec.shape)

# %%
from sklearn.naive_bayes import MultinomialNB,BernoulliNB, GaussianNB

# %%
mnb = MultinomialNB()
print(mnb)

# %%
mnb.fit(x_vec,y)

# %%
mnb.predict(xt_vec)

# %%
mnb.predict_proba(xt_vec)

# %%
mnb.score(x_vec,y)

# %%
bnb = BernoulliNB(binarize=0.0)

# %%
print(bnb)

# %%
bnb.fit(x_vec,y)

# %%
bnb.predict_proba(xt_vec)

# %%
bnb.predict(xt_vec)

# %%
bnb.score(x_vec,y)
