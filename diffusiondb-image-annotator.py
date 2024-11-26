import json
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torch.nn.functional import softmax

model_checkpoint = "VIT-Final-Finetune/checkpoint-3640"

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

# Load the pre-trained model and feature extractor (to process images)
model = AutoModelForImageClassification.from_pretrained(model_checkpoint)
processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)

# Load the full dataset (not streaming mode)
subset = '2m_first_1k'
dataset = load_dataset("poloclub/diffusiondb", name=subset, split="train")

# Define a collate function to preprocess a batch of images
def collate_fn(batch):
    images = [example['image'] for example in batch]
    inputs = processor(images=images, return_tensors="pt")  # Preprocess batch of images
    return inputs

# Create a DataLoader to handle batching (batch size of 8 here)
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize an empty list to store probabilities for each class
class_probs = {idx2label[str(i)]: [] for i in range(8)}  # 8 classes, will store all probabilities here
i = 0

# Process and predict on batches
for batch in tqdm(dataloader, desc="Processing batches"):
    # Pass the preprocessed batch of images to the model
    outputs = model(**batch)

    # Apply softmax to get probabilities
    logits = outputs.logits
    probs = softmax(logits, dim=-1)
    # maxes = torch.argmax(probs, dim=-1)
    #
    # for idx in maxes:
    #     class_probs[idx2label[str(idx.item())]] += 1
    # Accumulate probabilities for each class
    for img_probs in probs:  # Loop through each image in the batch
        for class_idx, prob in enumerate(img_probs):
            class_probs[idx2label[str(class_idx)]].append(prob.item())  # Store probability of each class

    # i += 1
    # if i > 1:  # Limit processing to two batches for testing purposes
    #     break

# Define the output path for the JSON file
class_probs_output_path = f'diffusiondb-{subset}-image-emotion-distributions.json'

# Write the class probabilities to a JSON file
with open(f'class-probs/{class_probs_output_path}', 'w') as json_file:
    json.dump(class_probs, json_file)

print(f"Class probabilities have been saved to {class_probs_output_path}")
