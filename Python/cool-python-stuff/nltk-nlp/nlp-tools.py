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
# !pip install nltk

# %%
paragraph = """
Narendra Damodardas Modi[a] (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress.

Modi was born and raised in Vadnagar, Bombay State (present-day Gujarat), where he completed his secondary education. He was introduced to the RSS at the age of eight. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he rose through the party hierarchy, becoming general secretary in 1998.[b] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat riots,[c] and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim.[4] A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him.[d] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[e]

In the 2014 Indian general election, Modi led the BJP to a parliamentary majority, the first for a party since 1984. His administration increased direct foreign investment, and reduced spending on healthcare, education, and social-welfare programmes. Modi began a high-profile sanitation campaign, and weakened or abolished environmental and labour laws. His demonetisation of banknotes in 2016 and introduction of the Goods and Services Tax in 2017 sparked controversy. Modi's administration launched the 2019 Balakot airstrike against an alleged terrorist training camp in Pakistan. The airstrike failed,[5][6] but the action had nationalist appeal.[7] Modi's party won the 2019 general election which followed. In its second term, his administration revoked the special status of Jammu and Kashmir, and introduced the Citizenship Amendment Act, prompting widespread protests, and spurring the 2020 Delhi riots in which Muslims were brutalised and killed by Hindu mobs.[8][9][10] Three controversial farm laws led to sit-ins by farmers across the country, eventually causing their formal repeal. Modi oversaw India's response to the COVID-19 pandemic, during which, according to the World Health Organization's estimates, 4.7 million Indians died.[11][12] In the 2024 general election, Modi's party lost its majority in the lower house of Parliament and formed a government leading the National Democratic Alliance coalition. Following a terrorist attack in Indian-administered Jammu and Kashmir, Modi presided over the 2025 India–Pakistan conflict, which resulted in a ceasefire.

Under Modi's tenure, India has experienced democratic backsliding and has shifted towards an authoritarian style of government, with a cult of personality centered around him.[f] As prime minister, he has received consistently high approval ratings within India. Modi has been described as engineering a political realignment towards right-wing politics. He remains a highly controversial figure domestically and internationally, over his Hindu nationalist beliefs and handling of the Gujarat riots, which have been cited as evidence of a majoritarian and exclusionary social agenda.[g]
"""

# %%
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# %%
nltk.download('punkt')
sentences = nltk.sent_tokenize(paragraph)

# %%
sentences

# %%
for i in sentences:
    print(i)

# %%
stemmer = PorterStemmer()

# %%
stemmer.stem("history")

# %%
from nltk.stem import WordNetLemmatizer

# %%
lemmatizer = WordNetLemmatizer()

# %%
lemmatizer.lemmatize("goes")

# %%
len(sentences)

# %%
import re

# %%
corpus = []
for i in sentences:
    corpus.append(re.sub('[^a-zA-Z]', ' ', i).lower())

# %%
corpus

# %%
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(stemmer.stem(word))

# %%
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(lemmatizer.lemmatize(word))

# %%
import re
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i]).lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)

# %%
#BagOfWords
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

# %%
X = cv.fit_transform(corpus)

# %%
cv.vocabulary_

# %%
X[0].toarray()

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True, ngram_range=(3, 3))

# %%
X = cv.fit_transform(corpus)
cv.vocabulary_

# %%
X[0].toarray()

# %%
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus)
