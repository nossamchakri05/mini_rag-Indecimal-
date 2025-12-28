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
        # Thresholds for 3-tier confidence
        high_conf_threshold = 0.8
        low_conf_threshold = 1.2
        
        # Increased retrieval depth
        results = vector_store.similarity_search_with_score(query, k=7)
        
        if not results:
            return "No relevant documents found."
            
        best_score = results[0][1]
        
        # 3-Tier Confidence Logic
        system_note = ""
        if best_score < high_conf_threshold:
            pass # High confidence, no note
        elif best_score < low_conf_threshold:
            system_note = f"SYSTEM NOTE: Moderate retrieval confidence (Score: {best_score:.2f}). Answer with disclaimer.\n\n"
        else:
            system_note = f"SYSTEM NOTE: Low retrieval confidence (Score: {best_score:.2f}). Use hedging language or refusal if unsure.\n\n"
            
        context_text = "\n\n".join(doc.page_content for doc, score in results)
        return f"{system_note}{context_text}"

    template = """You are an internal policy-aware assistant.

Rules:
1. Use ONLY the provided context.
2. If the answer is clearly present in the context, answer directly and confidently.
3. If the answer is partially present, answer with scope limitations.
4. Only say "not specified" if the information is completely absent.
5. Do NOT assume legal guarantees unless explicitly stated.
6. Prefer citing mechanisms, processes, and positioning over guarantees.
7. When lists exist, reproduce them faithfully.
8. CONFIDENCE OVERRIDE: If multiple retrieved chunks collectively answer the question, synthesize the answer instead of refusing.

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
    chain_with_context = RunnableParallel(
        {
            "context": RunnableLambda(retrieve_with_scores),
            "question": RunnablePassthrough()
        }
    ).assign(answer=prompt | llm | StrOutputParser())

    return chain_with_context
