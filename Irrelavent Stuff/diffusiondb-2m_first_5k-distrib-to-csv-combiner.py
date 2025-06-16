import json
import csv
import pandas as pd

# Load JSON files
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Transform JSON data to a format suitable for CSV
def transform_data(json_data):
    transformed = {"index": list(range(5000))}
    for emotion, values in json_data.items():
        transformed[emotion] = values
    return transformed

# Combine transformed data into a DataFrame
def create_dataframe(prompt_data, image_data):
    prompt_df = pd.DataFrame(transform_data(prompt_data))
    image_df = pd.DataFrame(transform_data(image_data))

    # Rename columns to distinguish between prompt and image data
    prompt_df = prompt_df.add_prefix('prompt_')
    image_df = image_df.add_prefix('image_')

    # Concatenate DataFrames on index
    combined_df = pd.concat([prompt_df, image_df], axis=1)
    combined_df.rename(columns={'prompt_index': 'index'}, inplace=True)
    return combined_df

# Save DataFrame to CSV
def save_to_csv(df, output_file):
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    prompt_file = "class-probs/diffusiondb-2m_first_5k-prompt-emotion-distributions.json"
    image_file = "class-probs/diffusiondb-2m_first_5k-image-emotion-distributions.json"
    output_file = "class-probs/diffusiondb-2m_first_5k-combined-emotion-distributions.csv"

    # Load JSON data
    prompt_data = load_json(prompt_file)
    image_data = load_json(image_file)

    # Create combined DataFrame
    combined_df = create_dataframe(prompt_data, image_data)

    # Save to CSV
    save_to_csv(combined_df, output_file)

    print(f"CSV file '{output_file}' has been created successfully.")