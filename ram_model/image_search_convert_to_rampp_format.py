import json
import os
from tqdm import tqdm

def flatten_dictionary(d):
    """Flatten a nested dictionary."""
    return [{key: value[key]} for key, value in d.items()]

def combine_and_flatten_json_files(directory):
    combined_data = {}

    # Read and combine JSON files
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.json'):
            if 'rampp' not in filename:
                with open(os.path.join(directory, filename), 'r') as file:
                    data = json.load(file)
                    combined_data.update(data)

    # Flatten the dictionary
    flattened_data = flatten_dictionary(combined_data)

    # Write to a new JSON file
    with open('rampp_overall.json', 'w') as file:
        json.dump(flattened_data, file, indent=4)

    print("Data combined and flattened successfully.")

if __name__ == "__main__":
    combine_and_flatten_json_files('../gpt_descriptions')