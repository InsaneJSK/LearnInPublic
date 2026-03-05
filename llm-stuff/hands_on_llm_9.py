# -*- coding: utf-8 -*-
"""Hands-on-llm-chap9
"""

!pip install cohere faiss-cpu rank_bm25 --quiet

from urllib.request import urlopen
from PIL import Image

puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"

image = Image.open(urlopen(puppy_path)).convert("RGB")

caption = "a puppy playing in the snow"

image

from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
model_id = "openai/clip-vit-base-patch32"
clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
clip_processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

inputs = clip_tokenizer(caption, return_tensors="pt")
inputs

clip_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

text_embedding= model.get_text_features(**inputs)
text_embedding.shape

processed_image = clip_processor(
    text = None, images=image, return_tensors='pt'
)['pixel_values']

processed_image.shape

import torch
import numpy as np
import matplotlib.pyplot as plt

img = processed_image.squeeze(0)
img = img.permute(*torch.arange(img.ndim - 1, -1, -1))
img = np.einsum("ijk->jik", img)

plt.imshow(img)
plt.axis("off")

image_embedding = model.get_image_features(processed_image)
image_embedding.shape

text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

text_embedding = text_embedding.detach().cpu().numpy()
image_embedding = image_embedding.detach().cpu().numpy()
score = np.dot(text_embedding, image_embedding.T)
score

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

car_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png"
image = Image.open(urlopen(car_path)).convert("RGB")
image

inputs = blip_processor(image, return_tensors='pt').to(device, torch.float16)
inputs['pixel_values'].shape

blip_processor.tokenizer

text = "Her vocalization was remarkably melodic"
token_ids = blip_processor(image, text=text, return_tensors='pt')
token_ids = token_ids.to(device, torch.float16)["input_ids"][0]
tokens = blip_processor.tokenizer.convert_ids_to_tokens(token_ids)[0]
tokens

tokens = [tokens.replace("Ġ", "_") for token in tokens]
tokens

image = Image.open(urlopen(car_path)).convert("RGB")
inputs = blip_processor(image, return_tensors='pt').to(device, torch.float16)
image

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens = True)
generated_text = generated_text[0].strip()
generated_text

url = "https://upload.wikimedia.org/wikipedia/commons/7/70/Rorschach_blot_01.jpg"
image = Image.open(urlopen(url)).convert("RGB")
inputs = blip_processor(image, return_tensors='pt').to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
generated_text = generated_text[0].strip()
generated_text

