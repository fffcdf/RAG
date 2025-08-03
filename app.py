from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing_extensions import TypedDict,List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import START, StateGraph
import streamlit as st

def load_docs():
    loader = PyMuPDFLoader('pandas_for_everyone.pdf')
    docs = loader.load()

    return docs

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    return all_splits

def vectorstore(all_splits,embeddings):
    vector_store=FAISS.from_documents(
        documents=all_splits,
        embedding=embeddings
    )
    return vector_store

def create_chat_prompt():
    system_template = """You are a helpful assistant. Use ONLY the information provided in the context below to answer the user's question.
    Do NOT use any prior knowledge or make assumptions.
    Keep your answer concise (no more than three sentences).
    End every answer with: "Thanks for asking!"
    If the answer is not found, say: 'I don't know.'"""

    human_template = """Context:
    {context}

    Question: {question}"""

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    return chat_prompt

class State(TypedDict,total=False):
    messages:List[BaseMessage]
    context:list
    answer:str

def retrieve(state: State):
    messages = state["messages"]
    user_question = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    retrieved_docs = vector_store.similarity_search(user_question)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = state["messages"]
    user_question = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)

    chat_messages = chat_prompt.format_messages(
        context=docs_content,
        question=user_question
    )

    response = llm.invoke(chat_messages)
    updated_messages=messages+[AIMessage(content=response.content)]
    return {"answer": response.content,
            'messages':updated_messages}

@st.cache_resource
def init_py():
    embeddings = OllamaEmbeddings(model='mxbai-embed-large:335m')
    llm = ChatOllama(model='phi')

    docs=load_docs()
    all_splits=split_docs(docs)

    vector_store=vectorstore(all_splits,embeddings)

    chat_prompt=create_chat_prompt()

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return {
        'embeddings':embeddings,
        'llm':llm,
        'vector_store':vector_store,
        'chat_prompt':chat_prompt,
        'graph':graph
    }

resources=init_py()

embeddings=resources['embeddings']
llm=resources['llm']
vector_store=resources['vector_store']
chat_prompt=resources['chat_prompt']
graph=resources['graph']

st.set_page_config(page_title="RAG", layout="wide")
st.title("RAG-chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Enter a question...")

if query:
    st.session_state.chat_history.append(HumanMessage(content=query))

    with st.spinner("Processing..."):
        outputs = graph.invoke({"messages": st.session_state.chat_history})
        answer = outputs["answer"]

        st.session_state.chat_history = outputs["messages"]

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if st.button("Clear the history"):
    st.session_state.chat_history = []
    st.rerun()

