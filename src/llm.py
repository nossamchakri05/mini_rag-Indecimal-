import os
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(model_type: str, model_name: str = None, api_key: str = None) -> BaseChatModel:
    """
    Returns an LLM instance based on the configuration.
    
    Args:
        model_type: 'openrouter' or 'local'
        model_name: Name of the model (e.g., 'openai/gpt-3.5-turbo', 'llama3')
        api_key: API key for OpenRouter
    """
    if model_type == 'openrouter':
        if not api_key:
            raise ValueError("API Key is required for OpenRouter.")
        
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name or "openai/gpt-3.5-turbo",
        )
    
    elif model_type == 'local':
        # Assuming Ollama for local execution as it's the easiest to integrate
        return Ollama(model=model_name or "llama3")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
