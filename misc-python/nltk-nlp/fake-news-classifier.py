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
import pandas as pd
df=pd.read_csv('fake_train.csv')

# %%
df.head()

# %%
df.isnull().sum()

# %%
###Drop Nan Values
df=df.dropna()

# %%
df.head()

# %%
## Get the Independent Features

X=df.drop('label',axis=1)
## Get the Dependent features
y=df['label']

# %%
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
### Vocabulary size
voc_size=5000

# %%
messages=X.copy()
messages['title'][1]

# %%
messages

# %%
messages.reset_index(inplace=True)

# %%
#Preprocessing
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer ##stemming purpose
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# %%
onehot_repr=[one_hot(words,voc_size)for words in corpus] 

# %%
#Embedding Representation
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length)
print(embedded_docs)

# %%
embedding_vector_features = 40  # Feature dimensions
voc_size = 5000                 # Example vocabulary size
sent_length = 100               # Example input sequence length

model = Sequential()
model.add(Embedding(input_dim=voc_size, output_dim=embedding_vector_features))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Explicitly build the model
model.build(input_shape=(None, sent_length))  # (batch_size, sequence_length)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# %%
import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)

# %%
X_final.shape,y_final.shape

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

# %%
### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

# %%
from tensorflow.keras.layers import Dropout
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# %%
### Dropout Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

# %%
y_pred=model.predict(X_test)

# %%
y_pred=np.where(y_pred > 0.6, 1,0) ##AUC ROC Curve

# %%
from sklearn.metrics import confusion_matrix

# %%
confusion_matrix(y_test,y_pred)

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
