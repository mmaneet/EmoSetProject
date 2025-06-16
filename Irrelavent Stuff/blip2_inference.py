import subprocess

from torchvision.datasets.EmoSet2 import EmoSet2
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import multiprocessing
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def normalize_image(image):
    """Normalize image to [0, 1] range"""
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model and processor
    print("Loading the model and processor...")
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()

    # Load the dataset
    print("Loading the dataset...")
    data_root = r"C:\Users\manee\EmoSet-118K-7"
    num_emotion_classes = 8

    dataset = EmoSet2(
        data_root=data_root,
        num_emotion_classes=num_emotion_classes,
        phase='val',
    )

    def custom_collate_fn(batch):
        # Create a batch of PIL images and other items
        images = [item['pil_image'] for item in batch]  # List of PIL images
        labels = [item['label'] for item in batch]  # List of labels
        image_paths = [item['image_path'] for item in batch]  # List of image paths

        # Return the batch as a dictionary
        return {'pil_image': images, 'label': labels, 'image_path': image_paths}

    # Then, use this custom collate function in your DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    # Process the dataset
    print("Generating captions...")
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            images = batch['pil_image']
            # from PIL import Image
            # images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in batch['pixel_value']]
            # plt.imshow(batch['pixel_value'][0].permute(1, 2, 0))
            image_paths = batch['image_path']  # Assuming 'image_path' is returned by EmoSet2
            labels = batch['label']

            # Debug: Print original image statistics
            # print(f"Original image shape: {images.shape}")
            # print(f"Original image min: {images.min().item()}, max: {images.max().item()}")

            # Normalize images if they're not in [0, 1] range
            # if images.min() < 0 or images.max() > 1:
            #     images = normalize_image(images)
            #     print(f"Normalized image min: {images.min().item()}, max: {images.max().item()}")

            try:
                # Process images
                print("Preprocessing now:")
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

                # Generate captions
                generated_ids = model.generate(
                                **inputs,
                                max_new_tokens= 20,
                            )

                captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

                for path, caption, label in zip(image_paths, captions, labels):
                    results.append({
                        'image_path': path,
                        'caption': caption,
                        'label': label
                    })
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Problematic image paths: {image_paths}")
                continue
            plt.show()
    # Convert to pandas DataFrame and save as CSV
    df = pd.DataFrame(results)
    csv_path = "image_captions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Captions generated and saved to '{csv_path}'")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
    subprocess.run([
        "python", "annotate.py",
        "--pretrained-folder", "C:/Users/manee/PycharmProjects/EmoSetProject/paletz",
        "--emotion-config",
        "C:/Users/manee/PycharmProjects/EmoSetProject/Demux-MEmo-master/emotion_configs/paletz.json",
        "--domain", "twitter",
        "--input-filename", "C:/Users/manee/PycharmProjects/EmoSetProject/image_captions.csv",
        "--input-format", "csv",
        "--out", "../emotion-annotations.jsonl",
        "--device", "cuda:0",
        "--text-column", "caption",
        "--id-column", "image_path"
    ], cwd=r"C:\Users\manee\PycharmProjects\EmoSetProject\Demux-MEmo-master", shell=True, check=True)

    subprocess.run(["python", "file_combiner.py"], cwd=r"C:\Users\manee\PycharmProjects\EmoSetProject", shell=True, check=True)