from fastapi import FastAPI
from .schemas import TitleRequest
from .model import MultiLingualE5Model, Specter2Model, SemanticModel


app = FastAPI()
model = SemanticModel()


# Send a Request body from the client (browser) to my API
@app.post("/find_similar")
async def find_similar(request: TitleRequest):
    result = model.find_most_similar(
        request.reference,
        request.other
    )
    return {"top_result": result}

