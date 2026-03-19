#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import torch
import pandas as pd

from pathlib import Path
from PIL import Image as PILImage
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoModelForImageClassification, AutoImageProcessor


in_file_path = sys.argv[1]
out_file_path = sys.argv[2]

model_checkpoint = "checkpoint-3640"
model = AutoModelForImageClassification.from_pretrained(model_checkpoint)
processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)

idx2label = {
    "0": "amusement",
    "1": "awe",
    "2": "contentment",
    "3": "excitement",
    "4": "anger",
    "5": "disgust",
    "6": "fear",
    "7": "sadness"
}

def predict_emotion(image_input):
    """
    Predict the emotion from an image using a fine-tuned Hugging Face model.

    Parameters:
        image_input (str or PIL.Image.Image): The image file path or a PIL image.

    Returns:
        tuple[int, dict[str, float]]: (predicted label index, probability map)
    """
    # Open the image if a file path is provided
    if isinstance(image_input, str):
        image = PILImage.open(image_input)
    else:
        image = image_input

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = softmax(logits, dim=-1)
    probs = probs.squeeze()

    max_idx = torch.argmax(probs, dim=-1).item()
    probs_map = {}

    for class_idx, prob in enumerate(probs):
        probs_map[idx2label[str(class_idx)]] = prob.item()
    # print("Image predicted.")
    return max_idx, probs_map


# In[ ]:


#from gpt_image_annotator import predict_emotion

img_emotion_cols = [
    'img_amusement', 'img_awe', 'img_contentment', 'img_excitement',
    'img_anger', 'img_disgust', 'img_fear', 'img_sadness'
]

def annotate_images(img_paths):

    updated = 0
    skipped = 0

    rows = []
    
    for image_path in tqdm(img_paths, total=len(img_paths), desc='Annotating images'):

        img_id = image_path.rpartition("/")[-1].rpartition(".")[0]
        
        if not isinstance(image_path, str) or not image_path.strip():
            skipped += 1
            continue

        resolved_path = Path(image_path)
        if not resolved_path.exists():
            skipped += 1
            continue

        try:
            _, probs = predict_emotion(str(resolved_path))
        except Exception as e:
            print(f'Error annotating {resolved_path}: {e}')
            skipped += 1
            continue

        rows.append({
            'img_id': img_id,
            'img_amusement': probs['amusement'],
            'img_awe': probs['awe'],
            'img_contentment': probs['contentment'],
            'img_excitement': probs['excitement'],
            'img_anger': probs['anger'],
            'img_disgust': probs['disgust'],
            'img_fear': probs['fear'],
            'img_sadness': probs['sadness'],
        })

        updated += 1

    df = pd.DataFrame(rows)

    print(f'Annotated {updated} images, skipped {skipped}.')
    return df


# In[ ]:


img_files = glob.glob(f"{in_file_path}/*.png")

df = annotate_images(img_files)
df.to_csv(out_file_path,index=False)
