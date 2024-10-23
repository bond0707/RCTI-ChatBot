import os
import g4f
import torch
import g4f.client
from g4f.Provider import RetryProvider, DDG, Koala, You
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class DataFormatter:
    def __init__(self):
        self.system_context = (
            "You are R.C. Bot, a friendly and knowledgeable chatbot for Ranchhodlal Chotalal Technical Institute "
            "(R.C. Technical Institute). Your primary role is to provide accurate and helpful information about the "
            "institute, including details about courses, admissions, campus facilities, events, and other related queries. "
            "You will be provided with a question and a paragraph for additional context. Your task is to generate an answer "
            "based on your knowledge and context. You should always be polite, approachable, and supportive in your responses."
            "If you realize that a question is not related to R.C. Technical Institute then you must tell the user to keep the"
            "questions related to R.C.T.I and not answer their original question."
        )
        self.messages = [
            {
                "role": "system",
                "content": self.system_context
            }
        ]
        model_path = "models/fine_tuned_models/RCTI_flan_t5_large_2e_qa"
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", clean_up_tokenization_spaces=True)
        self.tokenizer_max_length = 512
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="cuda",
        ).to("cuda")
        self.client = g4f.client.Client(
            provider=RetryProvider([DDG, Koala, You])
        )

    def update_messages(self, query, context_list):
        content = f"###Question:\n{query}\n\n###Context Paragraph:\n"
        for index, context in enumerate(context_list, 1):
            content += f"{index}. {context}\n"
        self.messages.append({"role": "user", "content": content})

    def get_model_completions(self, max_input_tokens=512, max_output_tokens=512):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(self.messages)
        tokenized_inputs = self.tokenizer(
            [msg["content"] for msg in self.messages],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_length
        ).to("cuda")

        tokens = self.model.generate(**tokenized_inputs, max_length=max_output_tokens)
        response = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return response

    def get_g4f_completions(self):
        chat = self.client.chat.completions.create(
            model="",
            messages=self.messages
        )
        return chat.choices[0].message.content


if __name__ == "__main__":
    query = "Who is the HOD of computer department?"
    context_list = [
        "Mr. Kamlesh N. Raval(KNR) is the H.O.D. of Computer Department.",
        "The vision of the Computer Department is to mould young and fresh minds into challenging computer professionals with ethical values, shaping them with upcoming technologies, and developing the ability to deal with real-world situations with skills and innovative ideas.",
        "To impart moral, ethical values, and interpersonal skills to the students.",
        "To impart necessary technical and professional skills among the students to make them employable and eligible for higher studies.",
        "To produce competent computer professionals by providing state-of-the-art training, hands-on experience, and skills for a practical environment."
    ]

    chatbot = DataFormatter()
    chatbot.update_messages(query, context_list)
    chatbot_response = chatbot.get_model_completions()
    print(f"Chatbot: {chatbot_response}")
