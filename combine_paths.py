import os
import json
import tiktoken  # Install with `pip install tiktoken`

# Define directories
INPUT_DIR = "paths"
OUTPUT_DIR = "combined_paths"
TOKEN_LIMIT = 3700

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load OpenAI's tokenizer (GPT-4)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Returns the number of tokens in a given text."""
    return len(tokenizer.encode(text))

# Collect all JSON files in the input directory
json_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])

# Initialize tracking variables
current_batch = []
current_tokens = 0
output_file_count = 0

for json_file in json_files:
    file_path = os.path.join(INPUT_DIR, json_file)

    try:
        # Read JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert JSON to string for token counting
        json_string = json.dumps(data, ensure_ascii=False, indent=4)
        json_token_count = count_tokens(json_string)

        # If adding this file exceeds token limit, save the current batch and start a new one
        if current_tokens + json_token_count > TOKEN_LIMIT and current_batch:
            # Write to a new output file
            output_file = os.path.join(OUTPUT_DIR, f"combined_{output_file_count}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(current_batch, f, ensure_ascii=False, indent=4)

            print(f"Saved {output_file} with {len(current_batch)} JSON objects.")
            
            # Reset batch
            current_batch = []
            current_tokens = 0
            output_file_count += 1

        # Add current file's JSON data to the batch
        current_batch.append(data)
        current_tokens += json_token_count

    except json.JSONDecodeError as e:
        print(f"Skipping {json_file} due to JSON decoding error: {e}")

# Save any remaining JSON objects in a final file
if current_batch:
    output_file = os.path.join(OUTPUT_DIR, f"combined_{output_file_count}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(current_batch, f, ensure_ascii=False, indent=4)
    
    print(f"Saved {output_file} with {len(current_batch)} JSON objects.")

print(f"Processed {len(json_files)} input files into {output_file_count + 1} output files in '{OUTPUT_DIR}'.")
