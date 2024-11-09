# R.C.T.I. chatbot

This project presents a chatbot for students and staff at R.C. Technical Institute. It uses an LLM trained with the custom R.C.T.I. dataset to answer questions related to the college, such as department heads, fee payment procedures, result viewing, class coordinators, and faculty contacts. The chatbot aims to make it easier to access important information and improve the overall user experience at the college. It is deployed [here.](https://rcti-chatbot.streamlit.app/ "RCTI chatbot")

## Project Features

1. **Data Retrieval:** Retrieves relevant information based on user prompts.
2. **Moderation:** Ensures responses are strictly related to RCTI topics.
3. **Text Generation:** Generates coherent and contextually accurate responses.
4. **Web UI:** Streamlit-based interface for easy user interaction.

## How it works?

This chatbot is essentially an expert system whose domain is R.C.T.I., its standard workflow is briefly explained below:

1. The user enters a prompt.
2. The prompt is then vectorized using `TfidfVectorizer()`
3. This vectorized prompt is then used to find the most similar questions in the dataset to the prompt using cosine similarity.
4. The top 5 results along with the prompt are then fed to the LLM.
5. This LLM will generate a coherent and human-like response for the given prompt.
6. This response is then displayed to the end user.

## The Knowledgebase

The project uses a custom-made dataset in a JSONLines file. Each object in the file has a "question" and an "answer" key. The question key follows a template to make it easy for the model to identify questions. Here’s an example of the format used:

```json
{   
"question": "### Question: “Your Question Here” \n\n\n###Answer: \n", 
 
"answer": "Your Answer Here" 
}
```

The data was manually scraped from the RCTI website and refined into questions. An LLM was used to generate these questions from the collected data automatically using a script.

## The Inference Engine

The project employs a two-step approach to handle user queries. Initially, a TF-IDF Vectorizer and cosine similarity are used to retrieve the five most relevant responses to the user's prompt from the dataset. These responses are then fed into a LLM which is fine-tuned specifically for answering RCTI-related questions. The LLM processes the provided context and generates a human like response, which is then presented to the user. I've used Google's ` flan-t5-large` pre-trained model and then fine tuned it on my custom-made RCTI dataset using the Transformers package provided by HuggingFace.
