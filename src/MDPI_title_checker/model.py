
import torch.nn.functional as F
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "intfloat/multilingual-e5-small"

class SimilarityModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
        
    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # Use pretrained model to compute vectors in embedding space for each input title
    def encode(self, texts):
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)
    
    def find_most_similar(self, reference, others):
        texts = ["query: " + reference] + ["passage: " + o for o in others]
        embeddings = self.encode(texts)
        
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        top_result_idx = scores.argmax().item()
        
        return others[top_result_idx]
        
