import json
import random
import spacy
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


class DataSelector():
    def __init__(self, knowledgebase_files, similarity_threshold=0.3):
        self.knowledgebase = self.load_knowledgebase(knowledgebase_files)
        self.nlp = self.load_spacy_model()
        self.vectorizer = TfidfVectorizer()
        self.similarity_threshold = similarity_threshold
        questions = [entry['question'] for entry in self.knowledgebase]
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)

    def load_spacy_model(self):
        try:
            return spacy.load('en_core_web_sm')
        except OSError:
            subprocess.run(
                ['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            return spacy.load('en_core_web_sm')

    def load_knowledgebase(self, file_paths):
        knowledgebase = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                for line in file:
                    knowledgebase.append(json.loads(line))
        return knowledgebase

    def get_5_closest_matches(self, query):
        matches = []
        tfidf_query = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(tfidf_query, self.tfidf_matrix)
        closest_indices = cosine_similarities.argsort()[0][-5:][::-1]
        for index in closest_indices:
            similarity = cosine_similarities[0, index]
            if similarity >= self.similarity_threshold:
                matches.append(self.knowledgebase[index]['answer'])
        avg_similarity = sum(
            cosine_similarities[0, idx] for idx in closest_indices) / 5
        return matches, avg_similarity


if __name__ == "__main__":
    knowledgebase_files = [
        "Data Pre-processing\\jsonlines_ds\\RCTI-Basic.jsonl",
    ]
    data_selector = DataSelector(knowledgebase_files)

    num_samples = 100  # Number of random samples to test
    average_similarities = []

    # Load the dataset to get random questions
    with open(knowledgebase_files[0], 'r') as file:
        dataset = [json.loads(line) for line in file]

    for _ in range(num_samples):
        random_entry = random.choice(dataset)
        query = random_entry['question']
        _, avg_similarity = data_selector.get_5_closest_matches(query)
        average_similarities.append(avg_similarity)

    print(f"Accuracy : {(sum(average_similarities)/len(average_similarities))*100}%")
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_samples), average_similarities, marker='o',
             linestyle='-', color='b', label='Average Cosine Similarity')
    plt.title('Average Cosine Similarity of Top 5 Matches')
    plt.xlabel('Sample Number')
    plt.ylabel('Average Cosine Similarity')
    plt.grid(True)
    plt.legend()
    plt.show()