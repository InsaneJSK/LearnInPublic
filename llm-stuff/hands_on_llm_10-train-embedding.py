# -*- coding: utf-8 -*-
"""Hands-on-llm-chap10-train-embedding
"""

from datasets import load_dataset

train_dataset = load_dataset(
    "glue", "mnli", split="train"
).select(range(50000))
train_dataset = train_dataset.remove_columns("idx")

train_dataset[2]

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("bert-base-uncased")

from sentence_transformers import losses
train_loss = losses.SoftmaxLoss(
    model = embedding_model,
    sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
    num_labels=3
)

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity = "cosine",
)

from sentence_transformers.training_args import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    output_dir = "base_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

from sentence_transformers.trainer import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()

evaluator(embedding_model)

train_dataset

from datasets import Dataset
mapping = {2:0, 1:0, 0:1}
train_dataset = Dataset.from_dict({
    "sentence1": train_dataset["premise"],
    "sentence2": train_dataset["hypothesis"],
    "label": [float(mapping[label]) for label in train_dataset["label"]]
})

val_sts

embedding_model = SentenceTransformer("bert-base-uncased")
train_loss = losses.CosineSimilarityLoss(model=embedding_model)
args = SentenceTransformerTrainingArguments(
    output_dir = "cosineloss_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

evaluator(embedding_model)

import random
from tqdm import tqdm
from datasets import Dataset, load_dataset

mnli = load_dataset("glue", "mnli", split="train").select(range(50000))
mnli = mnli.remove_columns("idx")
mnli = mnli.filter(lambda x: True if x["label"] == 0 else False)
train_dataset = {"anchor": [], "positive": [], "negative": []}
soft_negatives = list(mnli["hypothesis"])
random.shuffle(soft_negatives)
for row, soft_negative in tqdm(zip(mnli, soft_negatives)):
  train_dataset["anchor"].append(row["premise"])
  train_dataset["positive"].append(row["hypothesis"])
  train_dataset["negative"].append(soft_negative)
train_dataset = Dataset.from_dict(train_dataset)

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Create an embedding similarity evaluator for stsb
val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine"
)

from sentence_transformers import losses, SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Define model
embedding_model = SentenceTransformer('bert-base-uncased')

# Loss function
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="mnrloss_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

# Train model
trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

evaluator(embedding_model)
