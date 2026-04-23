from fastapi import FastAPI
from pydantic import BaseModel

import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


app = FastAPI()

class TitleRequest(BaseModel):
    reference: str
    other: list[str]

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Load model
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
    
# Send a Request body from the client (browser) to my API
@app.post("/find_similar")
async def find_similar(request: TitleRequest):

    reference_text = ['query: ' + request.reference]
    other_text = ['passage: ' + o for o in request.other]
    input_texts = reference_text + other_text
    
    # Use pretrained model to compute vectors in embedding space for each input title
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])


    # Use cosine distance for score
    embeddings = F.normalize(embeddings, p=2, dim=1) # always normalize the inputs for comparison
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    top_result_idx = scores.argmax().item()
    
    return {"top_result": request.other[top_result_idx]}


# 1) API endpoint that receives article titles as JSON object check
# 3  Use preprocessing tool to normalize titles?
# 4) Use pretrained model to compute vectors in embedding space for each input title
# 5) Use L2 distance or cosine distance in vector space 
# 6) Endpoint returns the title that is most similar to the reference title


# Example JSON Object input: 
# {
#   "reference": "Higgs boson in particle physics",
#   "other": [
#     "Best soup recipes", "Basel activities", "Particle physics at CERN"
#   ]
# }


# Output: {“top_result”: “Particle physics at CERN”}