import subprocess
from datasets import load_dataset
import pandas as pd
import json
#
# # Load the full dataset (not streaming mode)
subset = '2m_random_1k'
# dataset = load_dataset("poloclub/diffusiondb", name=subset, split="train")
#
# # Step 2: Extract the "image_id" and "prompt" columns
# prompts = dataset["prompt"]
# ids = [i for i in range(len(prompts))]
# # Step 3: Create a pandas DataFrame with the extracted columns
# df = pd.DataFrame({
#     "id": ids,
#     "prompt": prompts
# })
#
# id_captions_csv_file = 'C:/Users/manee/PycharmProjects/EmoSetProject/temp.csv'
# # Step 4: Save the DataFrame to a CSV file
# df.to_csv(id_captions_csv_file, index=False)
#
# print("CSV file created successfully!")
#
# # subprocess call to get text distribs from goemo
# subprocess.run([
#         "python", "annotate.py",
#         "--pretrained-folder", "C:/Users/manee/PycharmProjects/EmoSetProject/goemotions",
#         "--emotion-config",
#         "C:/Users/manee/PycharmProjects/EmoSetProject/Demux-MEmo-master/emotion_configs/goemotions.json",
#         "--domain", "twitter",
#         "--input-filename", id_captions_csv_file,
#         "--input-format", "csv",
#         "--out", "C:/Users/manee/PycharmProjects/EmoSetProject/diffusiondb-2m_first_1k-prompt-emotion-distributions-FULL.jsonl",
#         "--device", "cuda:0",
#         "--text-column", "prompt",
#         "--id-column", "id"
#     ], cwd=r"C:\Users\manee\PycharmProjects\EmoSetProject\Demux-MEmo-master", shell=True, check=True)

# List of target emotions to be aggregated
target_emotions = ["amusement", "surprise", "joy", "excitement", "anger", "disgust", "fear", "sadness"]

# Step 1: Pre-initialize the dictionary with empty lists for each target emotion
emotions_aggregated = {emotion: [] for emotion in target_emotions}

# Step 2: Open and read the .jsonl file
with open('diffusiondb-2m_random_1k-prompt-emotion-distributions-FULL.jsonl', 'r') as file:
    for line in file:
        # Parse each line as a JSON object
        data = json.loads(line)

        # Step 3: Iterate over the target emotions and add their scores if they exist in the current item
        for emotion in target_emotions:
            # Check if the target emotion is in the current item's emotions dictionary
            if emotion in data['emotions']:
                emotions_aggregated[emotion].append(data['emotions'][emotion])

class_probs_output_path = f'diffusiondb-{subset}-prompt-emotion-distributions.json'

with open(f'class-probs/{class_probs_output_path}', 'w') as json_file:
    json.dump(emotions_aggregated, json_file)

print(f'JSON file "{class_probs_output_path}" created successfully!')

