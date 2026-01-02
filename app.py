import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings # NEW for 2026
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 2026 CONFIGURATION ---
LLM_MODEL = "llama-3.2-3b-preview"
# DB_DIR must be a relative path in your GitHub repo (e.g., './chroma_db')
DB_DIR = "./Jan_2026_new_all-minilm_26122025" 

@st.cache_resource
def load_rag():
    # 1. Fetch API Key from Streamlit Secrets
    if "GROQ_API_KEY" not in st.secrets:
        st.error("Please add GROQ_API_KEY to your Streamlit Secrets.")
        st.stop()
    
    # 2. Use HuggingFace for FREE cloud-based embeddings
    # This replaces OllamaEmbeddings so it works on the internet
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Load your existing Chroma database
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 4. Initialize Groq LLM
    llm = ChatGroq(
        temperature=0.0, 
        model_name=LLM_MODEL, 
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    # ... [Rest of your retrieval & prompt logic remains the same] ...
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 7})
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    # 1. New Topic Detection Logic
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze history. If user asks a new question unrelated to previous nouns, treat it as a fresh start. Otherwise, link context."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_retriever = create_history_aware_retriever(llm, multi_query_retriever, context_q_prompt)

    # 2. COMPREHENSIVE PROMPT (Fixes short answers)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a detailed customer support AGI.
        
        STRICT INSTRUCTIONS:
        - Provide COMPREHENSIVE answers. Do not summarize if the context provides detailed steps.
        - List ALL available options, settings, and troubleshooting steps found in the context.
        - If the user uses layman language, map it to the technical documentation details.
        - Use ONLY the provided context. If not found, admit it.
         - dont fabricate answers
         - when giving answers, please dont add incorrect information which isnt in the context
         


        RESPONSE STRUCTURE:
        A: [Detailed Answer]
        
        Would you like help based on [Predicted Next Topic]? I can explain that next if you wish.
        
        CONTEXT:
        {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_retriever, qa_chain)

rag_bot = load_rag()

# --- CHAT INTERFACE WITH TIMESTAMPS ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages with timestamps
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.caption(f"🕒 {msg['time']}")
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about our services..."):
    # 1. Handle User Input
    ts_user = get_timestamp()
    st.session_state.messages.append({"role": "user", "content": prompt, "time": ts_user})
    with st.chat_message("user"):
        st.caption(f"🕒 {ts_user}")
        st.markdown(prompt)

    # 2. Handle Assistant Response
    with st.chat_message("assistant"):
        ts_bot = get_timestamp()
        st.caption(f"🕒 {ts_bot}")
        
        history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in rag_bot.stream({"input": prompt, "chat_history": history}):
            if "answer" in chunk:
                full_response += chunk["answer"]
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response, "time": ts_bot})
    
    # [Continue with history_retriever, qa_prompt, and qa_chain as in your original code]
    # return create_retrieval_chain(history_retriever, qa_chain)
