import openai
import requests
from torch.utils.data import DataLoader
from torchvision.datasets.EmoSet2 import EmoSet2
import pandas as pd
import os

# Load OpenAI API key from an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

if openai.api_key is None:
    raise ValueError("API key is missing. Please set the OPENAI_API_KEY environment variable.")

# Path to your image
image_path = r"C:\Users\manee\EmoSet-118K-7\image\amusement\amusement_00000.jpg"

data_root = r"C:\Users\manee\EmoSet-118K-7"
num_emotion_classes = 8

v_dataset = EmoSet2(
    data_root=data_root,
    num_emotion_classes=num_emotion_classes,
    phase='val'
)


def custom_collate_fn(batch):
    # Create a batch of PIL images and other items
    images = [item['pil_image'] for item in batch]  # List of PIL images
    labels = [item['label'] for item in batch]  # List of labels
    image_paths = [item['image_path'] for item in batch]  # List of image paths

    # Return the batch as a dictionary
    return {'pil_image': images, 'label': labels, 'image_path': image_paths}


# Then, use this custom collate function in your DataLoader
dataloader = DataLoader(v_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

prompt = "Generate a descriptive caption for this image that can help a model identify the emotion present."
results = []
image_counter = 0  # Counter to track the number of processed images
max_images = 200  # Limit the number of images to process

for batch in dataloader:
    images = batch['pil_image']
    image_paths = batch['image_path']
    labels = batch['label']

    # Break the loop if the limit has been reached
    if image_counter >= max_images:
        break

    captions = []
    for image in images:
        try:
            # Check if the counter exceeds the maximum number of images
            if image_counter >= max_images:
                break

            # Send image and prompt to the API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = response.json()
            captions.append(response['choices'][0]['message']['content'])
            image_counter += 1  # Increment the counter after processing each image
            print(image_counter)
        except Exception as e:
            print(e)
            print(response)
            continue

    for path, caption, label in zip(image_paths, captions, labels):
        results.append({
            'image_path': path,
            'caption': caption,
            'label': label
        })

# Convert to pandas DataFrame and save as CSV
df = pd.DataFrame(results)
csv_path = "chatgpt_captions.csv"
df.to_csv(csv_path, index=False)
print(f"Captions generated and saved to '{csv_path}'")

# Final message
print(f"Processed {image_counter} images and saved the results.")
