import streamlit as st
import ats_langchain
import ats_openai
import pinecone


model_names = [
                "ats_langchain_no_memory",
                "ats_openai",
               ]

modules = [
           ats_langchain,
           ats_openai,
          ]

module_dict = dict(zip(model_names, modules))

pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["ENV"])

index_names = pinecone.list_indexes()