import pinecone
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import streamlit as st
import os


index_name = None
openai_model = "gpt-3.5-turbo"

client = None
initial_messages = [{"role": "system", "content": "You are a customer support agent for a software product. You will be given an inquiry and some related past tickets and the corresponding comments. Please try to answer based on this information. Do not reference the tickets explicitly, but try to use their information to infer an answer about the inquiry. As the comments can vary in quality, please indicate if you are not sure about the answer."}]

messages = initial_messages

def init():
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["ENV"])
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    

def reset():
    global messages
    messages = initial_messages

def set_index(new_index_name):
    global index_name
    if index_name != new_index_name:
        index_name = new_index_name

        index = pinecone.Index(index_name)
        embed = OpenAIEmbeddings()
        global vectordb
        vectordb = Pinecone(index, embed, "text")

        global client
        client = OpenAI()
        # print(f"Updated index to {index_name}")


def query(prompt):
    if client is None:
        return "Please select a vector store in the sidebar."

    global messages

    search_string = prompt
    
    if len(messages) > 1:
        input = f"Here is the customer's inquiry {prompt}\n\n Give me a query to search the database for information so you can better answer this follow-up."
        messages.append({ "role": "user", "content": input })

        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=0.1
        )

        messages.pop()
        search_string = response.choices[0].message.content
        print(search_string)


    results = vectordb.similarity_search(search_string, k=2)
    example1 = results[0].page_content
    example2 = results[1].page_content


    input = f"Here is the customer's inquiry {prompt}\n\n Here are some related past tickets and the corresponding comments:\n\n Ticket 1:\n {example1}\n\n Ticket 2:\n {example2}."

    messages.append({ "role": "user", "content": input })

    response = client.chat.completions.create(
        model=openai_model,
        messages=messages,
        temperature=0.5
    )
    
    response_text = response.choices[0].message.content
    messages.append({ "role": "assistant", "content": response_text })

    return response_text
    