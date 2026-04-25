import json
from src.MDPI_title_checker.model import Specter2Model

def test_model():
    model = Specter2Model()

    with open("./tests/test_cases.json") as f:
        cases = json.load(f)

    for i, case in enumerate(cases):
        result = model.find_most_similar(
            case["reference"],
            case["other"]
        )
        print(f"\nTest {i+1}")
        print("Expected:", case["expected"])
        print("Got     :", result)
        assert result == case["expected"]