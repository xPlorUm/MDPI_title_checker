from fastapi import FastAPI

app = FastAPI()


# tells FastAPI that the function below is in charge of handling requests that go to the path / using a get operation
# when a user get this url then the function below is executed
@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}

# @app.get("/")
# async def read_item():
#     return {"item_id"}


# API endpoint that receives article titles as JSON object
# Use pretrained model to compute vectors in embedding space for each input title
# Use L2 distance or cosine distance in vector space 

# Endpoint returns the titlre that is most similar to the reference title




# Example JSON Object input: {“reference”: “Higgs boson in particle physics”, “other”:
#                [“Best soup recipes”, “Basel activities”, “Particle physics at
#                 CERN”]}

# Output: {“top_result”: “Particle physics at CERN”}