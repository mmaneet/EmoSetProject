import json
import csv

# Read the JSON file
def json_to_csv(filename):
    with open(f'{filename}.json', 'r') as json_file:
        data = json.load(json_file)

    # Prepare the CSV headers
    headers = [
        'macro_precision', 'macro_recall', 'macro_f1'
    ]

    # Add headers for each class
    for i in range(8):  # Assuming 8 classes based on the JSON structure
        headers.extend([f'class_{i}_precision', f'class_{i}_recall', f'class_{i}_f1'])

    # Prepare the row data
    row = [
        data['macro_precision'],
        data['macro_recall'],
        data['macro_f1']
    ]

    # Add data for each class
    for i in range(8):
        class_data = data['per_class_metrics'][f'class_{i}']
        row.extend([
            class_data['precision'],
            class_data['recall'],
            class_data['f1']
        ])

    # Write to CSV file
    with open(f'{filename}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerow(row)

    print(f"CSV file '{filename}.csv' has been created successfully.")

if __name__ == '__main__':
    json_to_csv('Goemo-Eval-Results-CHATGPT')
    json_to_csv('Goemo-Eval-Results-BLIP')
    json_to_csv('Paletz-Eval-Results-CHATGPT')
    json_to_csv('Paletz-Eval-Results-BLIP')
    json_to_csv('Semeval-Eval-Results-CHATGPT')
    json_to_csv('Semeval-Eval-Results-BLIP')