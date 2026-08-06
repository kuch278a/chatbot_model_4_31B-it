from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt, system_prompt=None, history=None, **kwargs):
        pass

class BaseEmbeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts):
        pass
        
    @abstractmethod
    def embed_query(self, text):
        pass
