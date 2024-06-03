from rag import RAG
import streamlit as st
import asyncio
st.session_state.clicked = False

vectorstore_created = False
flag = False

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return asyncio.get_event_loop()
    
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

@st.cache_resource(show_spinner = True)
def load_rag_pipeline(web_path):
    rag_pipeline=  RAG(web_path)
    return rag_pipeline

st.title("RAG Project")
st.subheader("This is end to end RAG pipeline")

web_path = st.sidebar.text_input("Enter website_url")
if web_path:
    rag_pipe = load_rag_pipeline(web_path)
    st.session_state.clicked = True

if st.session_state.clicked:
    st.subheader("Enter the question")
    question = st.text_input("Question")
    if question:
        answer, vs = rag_pipe.qa(question,vectorstore_created)
        vectorstore_created = vs
        st.subheader("Answer")
        # answer = rag_pipe.get_answer(question)
        st.write(answer)