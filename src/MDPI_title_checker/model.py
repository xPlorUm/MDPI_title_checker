
import torch.nn.functional as F
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
from .preprocessing import preprocess
from abc import ABC, abstractmethod

# MODEL_NAME = "intfloat/multilingual-e5-large"
MODEL_NAME = "allenai/specter2_base"

class SimilarityModel(ABC):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
    @abstractmethod
    def encode(input_text):
        pass
    
    def find_most_similar(self, reference, others): 
        normalized_text = preprocess([reference] + others)
        embeddings = self.encode(normalized_text)
        
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        top_result_idx = scores.argmax().item()
        
        return others[top_result_idx]
    
class MultiLingualE5Model(SimilarityModel):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(MODEL_NAME)
    
    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, input_text):
        texts = ["query: " + input_text[0]] + ["passage: " + o for o in input_text[1:]]
        
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        return F.normalize(embeddings, p=2, dim=1)
    
class Specter2Model(SimilarityModel):
    def __init__(self):
        super().__init__()
        self.model = AutoAdapterModel.from_pretrained(MODEL_NAME)
        self.model.load_adapter("allenai/specter2_proximity", source="hf", load_as="specter2", set_active=True)

    def encode(self, input_text):
        text_batch = [d + self.tokenizer.sep_token + '' for d in input_text] # can be extended with abstract in the future
        
        batch_dict = self.tokenizer(text_batch, padding=True, truncation=True,
                                        return_tensors="pt", return_token_type_ids=False, max_length=512)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return F.normalize(embeddings, p=2, dim=1)
        
