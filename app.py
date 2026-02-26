import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# --- SETUP ---
st.set_page_config(page_title="My Ollama Bot")
st.title("🤖 LangChain-Ollama Bot")

# 1. SET THE PERSONALITY (The "Tone")
# We define a 'System' instruction and a placeholder for your 'User' question.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's questions clearly."),
    ("user", "{question}")
])

# 2. SELECT THE BRAIN (The "Model")
# This tells LangChain to talk to your local Ollama engine.
llm = OllamaLLM(model="gemma2:2b")

# 3. CREATE THE CONNECTION (The "Chain")
# The '|' (pipe) symbol connects the Prompt to the LLM.
chain = prompt | llm

# --- CHAT INTERFACE ---

# 4. MEMORY (Session State)
# We create a list to store the chat so it doesn't disappear on refresh.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. HANDLING INPUT
if user_query := st.chat_input("Ask me anything..."):
    # Save and show your question
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Generate answer using the LangChain Chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # We send your query into the {question} slot of the prompt
            response = chain.invoke({"question": user_query})
            
            # Show and save the answer
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})