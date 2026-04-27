
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel
from abc import ABC, abstractmethod


class SimilarityModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    @abstractmethod
    def encode(self, input_text):
        pass
    
    def _encode_batch(self, input_text):
        text_batch = input_text
        
        batch_dict = self.tokenizer(
                                    text_batch,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt",
                                    return_token_type_ids=False, 
                                    max_length=512, 
                                    )
        
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        return F.normalize(embeddings, p=2, dim=1)
    
    def find_most_similar(self, reference, others):
        embeddings = self.encode([reference] + others)
        
        scores = (embeddings[:1] @ embeddings[1:].T) # cosine similarity
        top_result_idx = scores.argmax().item()
        
        return others[top_result_idx], scores
    
    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        denom = attention_mask.sum(dim=1).clamp(min=1)[..., None]
        return last_hidden.sum(dim=1) / denom
        

# Multilingual embedding model (good general semantic similarity performance, but slower)
class MultiLingualE5Model(SimilarityModel):
    def __init__(self):
        super().__init__("intfloat/multilingual-e5-large")
    
    def encode(self, input_text):
        # E5 expects "query:" and "passage:" prefixes
        texts = ["query: " + input_text[0]] + ["passage: " + o for o in input_text[1:]]
        return self._encode_batch(texts)
        
# Model trained on scientific papers (title/abstract similarity)
class Specter2Model(SimilarityModel):
    def __init__(self):
        super().__init__("allenai/specter2_base")

    def encode(self, input_text):
        return self._encode_batch(input_text)
    
# Lightweight sentence-transformer for semantic similarity, precise and fast just issues with abbreviations
class SemanticModel(SimilarityModel):
    def __init__(self):
        super().__init__("sentence-transformers/all-MiniLM-L6-v2")
        
    def encode(self, input_text):
        return self._encode_batch(input_text)

