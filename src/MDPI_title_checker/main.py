from fastapi import FastAPI
from .schemas import TitleRequest, TitleResponse
from .model import SimilarityModel


app = FastAPI()
model = SimilarityModel()


# Send a Request body from the client (browser) to my API
@app.post("/find_similar")
async def find_similar(request: TitleRequest):
    result = model.find_most_similar(
        request.reference,
        request.other
    )
    return {"top_result": result}


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