import os
import streamlit as st
from DataSelector import DataSelector
from DataFormatter import DataFormatter

if __name__ == "__main__":
    st.set_page_config(
        page_icon="🤖",
        page_title="RC Bot",
    )
    st.title("RC Bot - The RCTI Chatbot")
    
    selector = DataSelector([os.path.join(os.path.dirname(__file__), "Data Pre-processing", "jsonlines_ds", "RCTI-Basic.jsonl")])
    formatter = DataFormatter()
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something related to R.C. Technical Institute")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state["messages"].append({"role":"user", "content":prompt})
            matches = selector.get_5_closest_matches(prompt)
            formatter.update_messages(prompt, matches)

        with st.chat_message("assistant"):
            response = formatter.get_g4f_completions()
            st.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})            
