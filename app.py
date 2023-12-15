import streamlit as st
import model_selector
from time import sleep

st.title("Support Assistant")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Model selection
st.sidebar.title("Select a model:")
model_name = st.sidebar.selectbox("Model:", model_selector.model_names)

st.sidebar.title("Select a vector store for the model:")
index_name = st.sidebar.selectbox("Vector store:", model_selector.index_names)

# Model initialization
@st.cache_resource
def load_model(model_name):
    model = model_selector.module_dict[model_name]
    model.init()
    return model

model = load_model(model_name)
model.set_index(index_name)

# Clear chat history if model changed
if "model_name" not in st.session_state or model_name != st.session_state.model_name:
    st.session_state.messages = []
    st.session_state.model_name = model_name
    model.reset()


# Writing the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Reading user prompt
if prompt := st.chat_input("Ask me something"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({ "role": "user", "content": prompt })

    response = model.query(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)   

    st.session_state.messages.append({ "role": "assistant", "content": response })