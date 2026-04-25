# MDPI_title_checker
Assessment of MDPI including the implementation of a title similarity checker using FastAPI.

# How to install this project using poetry:
poetry install


# How to run this project:
1) poetry run uvicorn src.MDPI_title_checker.main:app --reload
2) 
    open http://127.0.0.1:8000/docs in Browser and use FastAPI Swagger UI to insert JSON
or 
    insert in a second PowerShell:
        Invoke-RestMethod -Method Post `
        -Uri "http://127.0.0.1:8000/find_similar" `
        -Headers @{ "Content-Type" = "application/json" } `
        -Body '{
            "reference": "Higgs boson in particle physics",
            "other": ["Best soup recipes", "Basel activities", "Particle physics at CERN"]
        }'
