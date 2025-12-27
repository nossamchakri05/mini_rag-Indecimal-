import streamlit as st
import os
# Imports moved to inside functions to avoid blocking UI startup
# from ingestion import load_documents, split_documents
# from embeddings import get_embedding_model
# from vector_store import create_vector_store, save_vector_store, load_vector_store
# from llm import get_llm
# from rag import create_rag_chain

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Mini RAG Assistant", layout="wide")

st.title("Construction Marketplace Assistant")

# Hardcoded Configuration
model_type = "openrouter"
model_name = "openai/gpt-3.5-turbo"
api_key = os.getenv("OPENROUTER_API_KEY")

def process_docs():
    with st.spinner("Processing documents..."):
        try:
            # Lazy imports
            from ingestion import load_documents, split_documents
            from embeddings import get_embedding_model
            from vector_store import create_vector_store, save_vector_store
            
            # 1. Load
            docs = load_documents("data")
            if not docs:
                st.error("No documents found in 'data/' folder.")
            else:
                st.info(f"Loaded {len(docs)} documents.")
                
                # 2. Split
                chunks = split_documents(docs)
                st.info(f"Created {len(chunks)} chunks.")
                
                # 3. Embed & Index
                embeddings = get_embedding_model()
                vector_store = create_vector_store(chunks, embeddings)
                save_vector_store(vector_store, "faiss_index")
                st.success("Indexing complete! Vector store saved.")
        except Exception as e:
            st.error(f"Error processing documents: {e}")

# Sidebar for Actions
with st.sidebar:
    st.header("Actions")
    
    # File Upload
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["pdf", "txt", "md"])
    
    if uploaded_files:
        if st.button("Save & Process Uploaded Files"):
            # Clear existing data
            data_dir = "data"
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    os.remove(os.path.join(data_dir, file))
            else:
                os.makedirs(data_dir)
            
            # Save new files
            for uploaded_file in uploaded_files:
                with open(os.path.join(data_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} files to data/ folder.")
            
            # Trigger processing
            process_docs()

    st.divider()
    
    # Manual Process Button (for existing files)
    if st.button("Reprocess Existing Files"):
        process_docs()

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about construction policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Lazy imports
            from embeddings import get_embedding_model
            from vector_store import load_vector_store
            from llm import get_llm
            from rag import create_rag_chain

            # Load resources
            embeddings = get_embedding_model()
            vector_store = load_vector_store("faiss_index", embeddings)
            
            if not vector_store:
                st.error("Vector store not found. Please process documents first.")
            else:
                llm = get_llm(model_type, model_name, api_key)
                rag_chain = create_rag_chain(vector_store, llm)
                
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(prompt)
                    
                    answer = response["answer"]
                    context = response["context"]
                    
                    if not answer:
                        st.warning("The model returned an empty response. This might be due to a strict prompt or lack of relevant context.")
                    
                    st.markdown(answer)
                    
                    with st.expander("Retrieved Context"):
                        st.markdown(context)
                        if not context:
                            st.warning("No relevant context found in documents.")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.error("Please check your API key (if using OpenRouter) or ensure Ollama is running (if using Local).")
