import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
import requests

# Load the DiffusionDB large_first_1k dataset from Hugging Face
dataset = load_dataset("poloclub/diffusiondb", name='large_first_1k', split="train")

# # Function to load an image from a URL
# def load_image(url):
#     return Image.open(requests.get(url, stream=True).raw)

# Display the first 10 images with their associated prompts
plt.figure(figsize=(20, 20))  # Set figure size
for i in range(10,20):
    # Get the image URL and the corresponding prompt
    image = dataset[i]['image']
    prompt = dataset[i]['prompt']

    # Load and display the image
    # image = load_image(image_url)
    plt.subplot(5, 2, i-10+1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(prompt)

plt.tight_layout()
plt.show()
