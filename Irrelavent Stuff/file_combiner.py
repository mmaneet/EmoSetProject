import csv
import json

# Define file paths
captions_csv = 'image_captions.csv'  # Change to your CSV file path
goemo_annotation_path = 'emotion-annotations-2.jsonl'  # Change to your JSONL file path
semeval_annotation_path = 'BLIP-semeval-annotations.jsonl'
paletz_annotation_path = 'emotion-annotations.jsonl'
output_file_path = 'BLIP-semeval-paletz-goemo.csv'  # Output CSV file path

print(f"Reading from {captions_csv} and {paletz_annotation_path}...")
# Read the CSV file into a dictionary with image_path as the key
csv_data = {}
with open(captions_csv, 'r', newline='', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        csv_data[row['image_path']] = row

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

goemo_label_2_idx = {
    "amusement": 0,
    "surprise": 1,
    "joy": 2,
    "excitement": 3,
    "anger": 4,
    "disgust": 5,
    "fear": 6,
    "sadness": 7
}

semeval_label_2_idx = {
    "joy": 0,
    "surprise": 1,
    "trust": 2,
    "optimism": 3,
    "anger": 4,
    "disgust": 5,
    "fear": 6,
    "sadness": 7
}

# reasoning for semeval maps
"""
amusement: joy
awe: surprise
anger: anger
contentment: trust
disgust: disgust
fear: fear
sadness: sadness
excitement: optimism

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
"""

target_emotions_paletz = ["amusement", "wonder", "happiness", "excitement", "anger", "disgust", "fear", "sadness"]
target_emotions_goemo = ["amusement", "surprise", "joy", "excitement", "anger", "disgust", "fear", "sadness"]
target_emotions_semeval = ["joy", "surprise", "trust", "optimism", "anger", "disgust", "fear", "sadness"]

# Process the JSONL file and find the argmax emotion
with (open(paletz_annotation_path, 'r', encoding='utf-8') as paletz_file, open(goemo_annotation_path, 'r', encoding='utf-8') as
goemo_file, open(semeval_annotation_path, 'r', encoding='utf-8') as semeval_file, open(output_file_path, 'w', newline='', encoding='utf-8') as output_file):
    # Define the fieldnames for the output CSV
    fieldnames = ['image_path', 'caption', 'emoset label', 'paletz pred', 'paletz pred id', 'goemo pred',
                  'goemo pred id', "semeval pred", "semeval pred id"]
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over each line (JSON object) in the JSONL file
    for line, line2, line3 in zip(paletz_file, goemo_file, semeval_file):
        paletz_item = json.loads(line)
        goemo_item = json.loads(line2)
        semeval_item = json.loads(line3)

        # Get the image path and emotion dictionary
        image_path = paletz_item['image_path']
        paletz_emotions = paletz_item['emotions']
        goemo_emotions = goemo_item['emotions']
        semeval_emotions = semeval_item['emotions']

        # Filter the emotions dictionary to only keep the target emotions
        filtered_paletz_emotions = {emotion: score for emotion, score in paletz_emotions.items() if
                             emotion.lower() in target_emotions_paletz}

        filtered_goemo_emotions = {emotion: score for emotion, score in goemo_emotions.items() if
                                    emotion.lower() in target_emotions_goemo}

        filtered_semeval_emotions = {emotion: score for emotion, score in semeval_emotions.items() if
                                    emotion.lower() in target_emotions_semeval}

        top_paletz_emotion = max(filtered_paletz_emotions, key=filtered_paletz_emotions.get)
        top_goemo_emotion = max(filtered_goemo_emotions, key=filtered_goemo_emotions.get)
        top_semeval_emotion = max(filtered_semeval_emotions, key=filtered_semeval_emotions.get)

        # Retrieve corresponding data from the CSV
        if image_path in csv_data:
            csv_row = csv_data[image_path]
            caption = csv_row['caption']
            emotion_id = csv_row['label']

            # Write the row to the output CSV
            writer.writerow({
                'image_path': image_path,
                'caption': caption,
                'emoset label': emotion_id,
                'paletz pred': top_paletz_emotion.lower(),
                'paletz pred id': paletz_label_2_idx[top_paletz_emotion.lower()],
                'goemo pred' : top_goemo_emotion.lower(),
                'goemo pred id' : goemo_label_2_idx[top_goemo_emotion.lower()],
                'semeval pred' : top_semeval_emotion.lower(),
                'semeval pred id' : semeval_label_2_idx[top_semeval_emotion.lower()]
            })
        else:
            print(f"Warning: Image path {image_path} in JSONL file not found in CSV.")

print(f"Files saved to {output_file_path} successfully.")