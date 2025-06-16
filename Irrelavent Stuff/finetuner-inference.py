import torch
from torchvision.datasets.EmoSet2 import EmoSet2
from transformers import logging, AutoModelForImageClassification, TrainingArguments, Trainer, pipeline
from torch.nn.functional import softmax
from evaluate import ImageClassificationEvaluator
model_checkpoint = "VIT-Final-Finetune/checkpoint-3640"  # pre-trained model from which to fine-tune
#150, 240, 270

label2idx = {
    "amusement": 0,
    "awe": 1,
    "contentment": 2,
    "excitement": 3,
    "anger": 4,
    "disgust": 5,
    "fear": 6,
    "sadness": 7
}

idx2label =  {
    "0": "amusement",
    "1": "awe",
    "2": "contentment",
    "3": "excitement",
    "4": "anger",
    "5": "disgust",
    "6": "fear",
    "7": "sadness"
}

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id = label2idx,
    id2label = idx2label,
)


dataset = EmoSet2(
    data_root=r"C:\Users\manee\EmoSet-118K-7",
    num_emotion_classes=8,
    phase="val"
)

print(dataset[0])

correct = 0
num_trials = 1
for i in range(num_trials):
    image = dataset[i]['pixel_value'].unsqueeze(0)
    actual_val = dataset[i]['label']
    map = {'pixel_values' : image}
    outputs = model(**map)
    # Extract logits
    logits = outputs.logits

    # Apply softmax to get probabilities
    print(logits)
    probs = softmax(logits, dim=-1)
    print(probs)
    max_prob_index = torch.argmax(probs, dim=-1)

    if actual_val == max_prob_index.item():
        correct += 1
    # Print the index and the corresponding label
    print(f"Image {i}: Max Probability Index: {max_prob_index.item()}, Label: {idx2label[str(max_prob_index.item())]}, Actual Label: {idx2label[str(actual_val)]}")

print(f"Num Correct: {correct} || Percent Correct: {correct/num_trials :.3%}")