import json

import evaluate
from torch.utils.data import DataLoader
from torchvision.datasets.EmoSet2 import EmoSet2
from datasets import Dataset as HFDataset, DatasetDict, Features, ClassLabel, Image
from sklearn.metrics import classification_report
from transformers import logging, AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

model_checkpoint = "convnextv2-tiny-1k-224-emoset-finetune/checkpoint-3690"  # pre-trained model from which to fine-tune
batch_size = 16  # batch size for training and evaluation

data_root = r"C:\Users\manee\EmoSet-118K-7"
num_emotion_classes = 8
print(data_root)

t_dataset = EmoSet2(
    data_root=data_root,
    num_emotion_classes=num_emotion_classes,
    phase='train',
)

v_dataset = EmoSet2(
    data_root=data_root,
    num_emotion_classes=num_emotion_classes,
    phase='val',
)

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

print("asdkaljf")
# train_dict = [t_dataset[i] for i in range(len(t_dataset))]
# val_dict = [v_dataset[i] for i in range(len(v_dataset))]

print("kalsdjalks")
# train_ds = HFDataset.from_list(train_dict)
# val_ds = HFDataset.from_list(val_dict)

print("talkjasf ??")

# print(train_ds)
# print(val_ds)
# print(val_ds[0])

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2idx,
    id2label=idx2label,
)

# Freeze all parameters except the classification head
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classification head parameters
for param in model.classifier.parameters():
    param.requires_grad = True

model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"VIT-Final-Finetune-Inference",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=130,
    save_steps=130,
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=20,
    logging_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # eval_on_start = True
)

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)

    # Compute macro-average metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

    # Create a dictionary with metrics for each class
    class_metrics = {}
    for i in range(len(precision)):  # This will work for any number of classes
        class_metrics[f'class_{i}'] = {
            'title': idx2label[str(i)],
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }

    # Combine all metrics
    results = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class_metrics': class_metrics
    }

    return results


import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_value"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=t_dataset,
    eval_dataset=v_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.evaluate()
print(train_results)

# Dump results to JSON file
json_filename = "ConvNeXT-Eval-Results.json"
with open(json_filename, 'w') as json_file:
    json.dump(train_results, json_file, indent=4)
print(f"Evaluation results saved to {json_filename}")

# rest is optional but nice to have
# trainer.save_model()
# trainer.log_metrics("train", train_results)
# trainer.save_metrics("train", train_results)
# trainer.save_state()
