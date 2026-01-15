"""
Embedding utilities - Generate vector embeddings using LiteLLM
Refactored to support API-based embeddings (Gemini, OpenAI) via LiteLLM
"""
from typing import List, Union, Optional
from litellm import embedding
import config

class EmbeddingModel:
    """
    Embedding model using LiteLLM (supports Gemini, OpenAI, etc.)
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.api_key = config.OPENAI_API_KEY
        
        print(f"Loading embedding model: {self.model_name}")
        
        # Determine dimension based on known models to avoid initial API call costs/latency
        if "text-embedding-004" in self.model_name:
            self.dimension = 768
        elif "ada-002" in self.model_name or "3-small" in self.model_name:
            self.dimension = 1536
        elif "qwen" in self.model_name.lower():
             self.dimension = 1024 # Default for Qwen3-0.6B as per original config
        else:
            # Fallback: make a dummy call to check dimension
            try:
                print("Detecting embedding dimension...")
                dummy = self.encode_single("test")
                self.dimension = len(dummy)
                print(f"Detected dimension: {self.dimension}")
            except Exception as e:
                print(f"Error detecting dimension: {e}")
                self.dimension = 768 # Fallback default
                
    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts into embeddings
        """
        if not texts:
            return []
            
        try:
            # LiteLLM handles batching logic for many providers, 
            # but standard embedding calls take a list.
            response = embedding(
                model=self.model_name,
                input=texts,
                api_key=self.api_key
            )
            # Ensure correct order
            data = response['data']
            if hasattr(data, 'sort'): # generic response object might be dict or object
                 data.sort(key=lambda x: x['index'])
            elif isinstance(data, list):
                 data = sorted(data, key=lambda x: x['index'])
                 
            return [item['embedding'] for item in data]
        except Exception as e:
            print(f"Embedding error: {e}")
            raise e

    def encode_single(self, text: str, is_query: bool = False) -> List[float]:
        """
        Encode a single text
        """
        embeddings = self.encode_documents([text])
        if not embeddings:
            raise ValueError("Failed to generate embedding")
        return embeddings[0]
        
    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension