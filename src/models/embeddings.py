import torch
from transformers import AutoTokenizer, AutoModel
from .base import BaseEmbeddings

class LocalMultilingualEmbeddings(BaseEmbeddings):
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        print(f"Loading local embedding model from: {model_path} onto {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        self.model.eval()

    def embed_documents(self, texts):
        if not texts:
            return []
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy().tolist()
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]
