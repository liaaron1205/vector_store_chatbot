import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import streamlit as st
import os


index_name = None
openai_model = "gpt-3.5-turbo"

qa = None


def init():
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["ENV"])
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    

def reset():
    pass

def set_index(new_index_name):
    global index_name
    if index_name != new_index_name:
        index_name = new_index_name

        index = pinecone.Index(index_name)
        embed = OpenAIEmbeddings()
        vectordb = Pinecone(index, embed, "text")
        llm = ChatOpenAI(
            openai_api_key = st.secrets["OPENAI_API_KEY"],
            model_name = openai_model,
            temperature = 0.7,
        )
        global qa
        qa = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever()
        )
        # print(f"Updated index to {index_name}")

    

def query(prompt):
    if qa is None:
        return "Please select a vector store in the sidebar."
    
    return qa.run(prompt)
    