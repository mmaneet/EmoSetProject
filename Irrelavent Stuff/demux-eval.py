import json

from sklearn.metrics import precision_recall_fscore_support

# TODO: Generate emotion annotations for GoEmotions (run the following script:

# TODO: Generate a csv file that has the labels, the top emotion from the relavent 8 for paletz, and the top emotion
#  from the relavent 8 for goemotions

# TODO: evaluate the CSV file, with the predictions as paletz_label_2_idx[predictons]

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

paletz_label_2_idx = {
    "amusement": 0,
    "wonder": 1,
    "happiness": 2,
    "excitement": 3,
    "anger": 4,
    "disgust": 5,
    "fear": 6,
    "sadness": 7
}

def compute_metrics(predictions, labels):
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

import csv

def extract_columns_from_csv(file_path):
    # Initialize empty lists to store column data
    emoset_labels = []
    goemo_pred_ids = []
    paletz_pred_ids = []
    semeval_pred_ids = []

    # Open and read the CSV file
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)

        # Iterate over each row and extract data from the specified columns
        for row in csv_reader:
            emoset_labels.append(row['emoset label'])
            goemo_pred_ids.append(row['goemo pred id'])
            paletz_pred_ids.append(row['paletz pred id'])
            semeval_pred_ids.append(row['semeval pred id'])

    # Return the extracted data as lists
    return emoset_labels, goemo_pred_ids, paletz_pred_ids, semeval_pred_ids

# Example usage
file_path = 'chatgpt-semeval-paletz-goemo.csv'  # Replace with your actual file path
emoset_labels, goemo_pred_ids, paletz_pred_ids, semeval_pred_ids = extract_columns_from_csv(file_path)
goemo_results = compute_metrics(goemo_pred_ids, emoset_labels)
paletz_results = compute_metrics(paletz_pred_ids, emoset_labels)
semeval_results = compute_metrics(semeval_pred_ids, emoset_labels)

import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, np.int64):  # If the object is a NumPy int64, convert to Python int
        return int(obj)
    elif isinstance(obj, np.float64):  # Convert np.float64 to native Python float
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

json_filename = "Goemo-Eval-Results-CHATGPT.json"
with open(json_filename, 'w') as json_file:
    json.dump(goemo_results, json_file, indent=4, default=convert_to_serializable)
print(f"Evaluation results saved to {json_filename}")

json_filename = "Paletz-Eval-Results-CHATGPT.json"
with open(json_filename, 'w') as json_file:
    json.dump(compute_metrics(paletz_pred_ids, emoset_labels), json_file, indent=4, default=convert_to_serializable)
print(f"Evaluation results saved to {json_filename}")

json_filename = "Semeval-Eval-Results-CHATGPT.json"
with open(json_filename, 'w') as json_file:
    json.dump(compute_metrics(semeval_pred_ids, emoset_labels), json_file, indent=4, default=convert_to_serializable)
print(f"Evaluation results saved to {json_filename}")