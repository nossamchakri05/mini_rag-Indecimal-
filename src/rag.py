from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.runnables import RunnableLambda

def create_rag_chain(vector_store: VectorStore, llm: BaseChatModel):
    """Creates a RAG chain with guardrails and confidence scoring."""
    
    def retrieve_with_scores(query: str):
        """Retrieves documents and checks confidence scores."""
        # FAISS default is L2 distance. Lower is better.
        # Threshold: > 1.0 implies weak similarity for normalized vectors.
        threshold = 1.0
        
        results = vector_store.similarity_search_with_score(query, k=3)
        
        docs = []
        low_confidence = False
        
        if not results:
            return "No relevant documents found."
            
        best_score = results[0][1]
        if best_score > threshold:
            low_confidence = True
            
        context_text = "\n\n".join(doc.page_content for doc, score in results)
        
        if low_confidence:
            context_text = f"SYSTEM NOTE: Low retrieval confidence (Score: {best_score:.2f}). Use hedging language.\n\n{context_text}"
            
        return context_text

    template = """You are a strict assistant for a construction marketplace.
Answer the question based ONLY on the following context.

GUARDRAILS:
1. If the answer is not explicitly stated, you MUST say "The provided documents do not explicitly state [concept]" or "Not specified in this document".
2. ALLOWED phrases: "Positioned as", "Described as", "Subject to contract", "Not specified in this document".
3. PROHIBITED phrases (unless explicitly in text): "Always", "Guaranteed", "Legal documentation", "Contractually defined".
4. If the context mentions "Low retrieval confidence", use hedging language (e.g., "It appears...", "The documents suggest...").

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        RunnableParallel(
            {
                "context": RunnableLambda(retrieve_with_scores),
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Return a chain that returns a dict with 'answer' and 'context' to match app.py expectation
    # We need to reconstruct the context for display, so we run retrieval separately or use a different structure.
    # To keep it compatible with app.py which expects {'answer': ..., 'context': ...}, we need to adjust.
    
    chain_with_context = RunnableParallel(
        {
            "context": RunnableLambda(retrieve_with_scores),
            "question": RunnablePassthrough()
        }
    ).assign(answer=prompt | llm | StrOutputParser())

    return chain_with_context
