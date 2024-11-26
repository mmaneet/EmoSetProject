import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# Use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and processor
print("Loading model and processor...")
model_name = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
model.to(device)
model.eval()
torch.set_grad_enabled(False)

# Print model configuration
print("Model config:", model.config)

# Load and process image
image_path = r"C:\Users\manee\EmoSet-118K-7\image\amusement\amusement_00000.jpg"
image = Image.open(image_path).convert('RGB')

# Process image and generate caption
print("Generating caption...")
inputs = processor(images=image, return_tensors="pt").to(device)
prompt = "Describe this image:"
input_ids = processor(text=prompt, return_tensors="pt").input_ids.to(device)

# Print input shapes for debugging
print("Image input shape:", inputs['pixel_values'].shape)
print("Text input shape:", input_ids.shape)
print("Input pixel values range:", inputs['pixel_values'].min().paletz_item(), "-", inputs['pixel_values'].max().paletz_item())

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        input_ids=input_ids,
        max_new_tokens=100,
        num_beams=5,
        do_sample=True,
        temperature=0.7,
        min_length=10,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True
    )

generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print("Caption:", generated_text)

# Print some debug information
print("Output shape:", outputs.shape)
print("Raw output:", outputs)

# Print the tokenizer's vocabulary size
print("Vocabulary size:", len(processor.tokenizer))

# Print the first few tokens of the output
print("First 10 output tokens:", outputs[0][:10])