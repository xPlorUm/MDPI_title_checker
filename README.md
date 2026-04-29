# MDPI_title_checker
Assessment of MDPI including the implementation of a title similarity checker using FastAPI.

# Task description:

1) API endpoint that receives article titles as JSON object check
2) (optional) Use preprocessing tool to normalize titles
3) Use pretrained model to compute vectors in embedding space for each input title
4) Use L2 distance or cosine distance in vector space 
5) Endpoint returns the title that is most similar to the reference title

## Example Input: 
```
{
  "reference": "Higgs boson in particle physics",
  "other": [
    "Best soup recipes", "Basel activities", "Particle physics at CERN"
  ]
}
```

## Example Output:
``` 
{
  “top_result”: “Particle physics at CERN”
}
```

# Installation using Poetry:
```
poetry install
```

# Running the API
```
poetry run uvicorn src.MDPI_title_checker.main:app --reload
```

open the browser:  
  http://127.0.0.1:8000/docs/

Use the FastAPI Swagger UI to test the endpoint.

# Sending Requests
## Windows (Powershell)
```
Invoke-RestMethod -Method Post `
-Uri "http://127.0.0.1:8000/find_similar" `
-Headers @{ "Content-Type" = "application/json" } `
-Body '{
    "reference": "Higgs boson in particle physics",
    "other": ["Best soup recipes", "Basel activities", "Particle physics at CERN"]
}'
```

## Linux/MacOS
```
curl -X POST "http://127.0.0.1:8000/find_similar" \
     -H "Content-Type: application/json" \
     -d '{
           "reference": "Higgs boson in particle physics",
           "other": [
             "Best soup recipes",
             "Basel activities",
             "Particle physics at CERN"
           ]
         }'
```
# Tests
Tests are generated using LLM.

The current implementation does not involve a correct transformation of abbreviations to embedding space.

## Running Tests
```
poetry run pytest
```
