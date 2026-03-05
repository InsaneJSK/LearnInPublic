# !pip install transformers>=4.40.1 accelerate>=0.27.2 sentence-transformers --q
# !pip install -U datasets --q

from datasets import load_dataset

data = load_dataset("rotten_tomatoes")
data

data["train"][0, -1]

from transformers import pipeline

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
pipe = pipeline(
    model = model_path,
    tokenizer = model_path,
    return_all_scores = True,
    device = "cuda:0"
)

import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

y_pred = []
for out in tqdm(pipe(KeyDataset(data["test"], "text")), total = len(data["test"])):
  neg_score = out[0]["score"]
  pos_score = out[2]["score"]
  assignment = np.argmax([neg_score, pos_score])
  y_pred.append(assignment)

from sklearn.metrics import classification_report

def evaluate(y_true, y_pred):
  print(classification_report(y_true, y_pred, target_names = ["Negative Reviews", "Positive Reviews"]))

evaluate(data["test"]["label"], y_pred)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

train_embeddings.shape

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state = 42)
clf.fit(train_embeddings, data["train"]["label"])

y_pred = clf.predict(test_embeddings)
evaluate(data["test"]["label"], y_pred)

label_embeddings = model.encode(["A Negative Review", "A Positive Review"])

from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis = 1)
evaluate(data["test"]["label"], y_pred)

