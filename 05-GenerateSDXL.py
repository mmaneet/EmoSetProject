import sys
import glob
import json
import pandas as pd

prompt_json = sys.argv[1]
gpt_image_path = sys.argv[2]
output_path = sys.argv[3]


img_size = 1024
n_steps = 64
guidance_scale = 8
num_images = 1


prompt_id_pairs = []
with open(prompt_json, "r") as in_file:
    for line in in_file:
        batch_line = json.loads(line)
        prompt_id_pairs.append({
            "custom_id": batch_line["custom_id"],
            "prompt": batch_line["body"]["prompt"],
        })

prompt_id_pairs_df = pd.DataFrame(prompt_id_pairs)

gpt_gened_images = [f.rpartition("/")[-1].replace(".png", "") for f in glob.glob(f"{gpt_image_path}/*.png")]
print("Have %d images" % len(gpt_gened_images))

rel_pid_pairs_df = prompt_id_pairs_df[prompt_id_pairs_df["custom_id"].isin(gpt_gened_images)]
print("Of ", prompt_id_pairs_df.shape[0], "prompts, we have", rel_pid_pairs_df.shape[0], "images")

from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True
)
pipeline = pipeline.to("cuda:0")

for _,row in rel_pid_pairs_df.iterrows():
    filename = row["custom_id"] + ".png"
    prompt = row["prompt"]
    
    images = pipeline(
        prompt,
        height=img_size, 
        width=img_size, 
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
    ).images
    
    for idx,image in enumerate(images):
        image.save(f"{output_path}/%s" % (filename, ))
        print(f"{output_path}/%s" % (filename, ))