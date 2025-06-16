import matplotlib.pyplot as plt
import json

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

# Define the output path for the JSON file
class_probs_file_path = "diffusiondb-2m_first_1k-image-emotion-distributions.json"

# Write the class probabilities to a JSON file
with open(f'class-probs/{class_probs_file_path}', 'r') as json_file:
    class_probs = json.load(json_file)

# Plot histograms for each class
plt.figure(figsize=(14, 10))  # Set figure size
for classname, probs in class_probs.items():
    plt.hist(probs, bins=50, alpha=0.5, label=f'{classname}')

plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title(class_probs_file_path[:-5])
plt.legend()

# Save the plot as a PNG file
plt.savefig(f"{class_probs_file_path[:-5]}.png")

# Show the plot
plt.show()