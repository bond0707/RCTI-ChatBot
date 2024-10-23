import os
import ast
import g4f
import json
import asyncio
import g4f.client
from pprint import pprint
from g4f.Provider import RetryProvider, DDG, You, Koala

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def read_jsonl_data(pathlist):
    dataset = []
    for filepath in pathlist:
        dataset += [json_obj for json_obj in open(filepath, "r").readlines()]
    return dataset

def feed_dataset_to_LLM(dataset):
    improved_dataset_subset = []
    client = g4f.client.Client(provider=RetryProvider([DDG, Koala, You]))
    messages = [{"role": "system", "content": "You are an ML developer helping in making a question-answering chatbot. You will be provided a list containing data in jsonlines format. Each json object will have 2 keys 'question' and 'answer'. Your task is to generate new questions which have the same answer as the original question for every object. You will return the data in a PYTHON LIST without any name or assignment, just using the braces '[' and ']', where each element is a JSONLINES object. Also write each newline charater with two slashes so that json.loads() methods works properly"}]

    while len(dataset) > 0:
        if len(dataset) < 10:
            dataset_subset = [dataset.pop(0) for i in range(len(dataset))]
        else:
            dataset_subset = [dataset.pop(0) for i in range(10)]

        messages.append({"role": "user", "content": "".join(dataset_subset)})
        print("\nFetching improved data.")
        resp = client.chat.completions.create(
            model="",
            messages=messages
        )
        improved_dataset_subset = eval(resp.choices[0].message.content)
        print("Improved data fetched successfully.\nWriting it into a file...")          

        with open("improved_dataset.jsonl", "a+") as f:
            for i in improved_dataset_subset:
                json.dump(i, f)
                f.write("\n")
        print("Improved data appended to 'improved_dataset.jsonl' successfully!")

        dataset_subset = []
        improved_dataset_subset = []
    
if __name__ == "__main__":
    DATASET_PATH_LIST = [os.path.join(os.path.dirname(__file__), "jsonlines_ds", "RCTI-Basic.jsonl")]
    data = read_jsonl_data(DATASET_PATH_LIST)
    feed_dataset_to_LLM(data)
    print("Done!")