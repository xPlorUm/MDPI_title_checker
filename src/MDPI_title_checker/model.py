
import torch.nn.functional as F
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

MODEL_NAME = "intfloat/multilingual-e5-small"

class SimilarityModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
        
    def preprocess(self, texts):
        return [self.normalize(text) for text in texts]
        
    def normalize(self, string):
        # convert to lower case
        no_number_string = string.lower()
        
        # remove punctuations
        no_punc_string = re.sub(r'[^\w\s]', '', no_number_string)
        
        # remove whitespace
        no_wspace_string = no_punc_string.strip()
        
        # remove stopwords
        lst_string = [no_wspace_string][0].split()
        no_stpwords_string = ""
        for i in lst_string:
            if not i in stop_words:
                no_stpwords_string += i + ' '
        no_stpwords_string = no_stpwords_string[:-1] # remove last space
        
        return no_stpwords_string
        
        
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
        normalized_text = self.preprocess([reference] + others)
        texts = ["query: " + normalized_text[0]] + ["passage: " + o for o in normalized_text[1:]]
        embeddings = self.encode(texts)
        
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        top_result_idx = scores.argmax().item()
        
        return others[top_result_idx]
        
