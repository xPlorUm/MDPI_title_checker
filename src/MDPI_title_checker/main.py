from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class TitleRequest(BaseModel):
    reference: str
    other: list[str]

# @app.get("/")
async def root():
    return {"item_id"}

# when a user calls this url then the function below is executed
@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}

# Send a Request body from the client (browser) to my API
@app.post("/find_similar")
async def find_similar(request: TitleRequest):
    return request.reference


# 1) API endpoint that receives article titles as JSON object check
# 2) Use pretrained model to compute vectors in embedding space for each input title
# 3) Use L2 distance or cosine distance in vector space 
# 4) Endpoint returns the title that is most similar to the reference title




# Example JSON Object input: {“reference”: “Higgs boson in particle physics”, “other”:
#                [“Best soup recipes”, “Basel activities”, “Particle physics at
#                 CERN”]}

# Output: {“top_result”: “Particle physics at CERN”}