import os
import json
import jsonlines

JSON_PATH = os.path.join(os.path.dirname(__file__), "json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "jsonlines_ds")
os.makedirs(OUTPUT_PATH, exist_ok=True)

for file in os.listdir(JSON_PATH):
    print("Making .jsonl file from :", file)
    with open(os.path.join(JSON_PATH, file), "r") as f:
        data = json.load(f)
        
        for i in data:
            i["question"] = "### Question:\n" + i["question"] + "\n\n\n### Answer: \n"
        
        output_file = os.path.join(OUTPUT_PATH, file.replace('.json', '.jsonl'))
        with jsonlines.open(output_file, 'w') as writer:
            for i in data:
                writer.write(i)

print("All files converted sucessfully!")