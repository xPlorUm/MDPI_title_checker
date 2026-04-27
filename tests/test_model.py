import json
import pytest
from src.MDPI_title_checker.model import Specter2Model, SemanticModel, MultiLingualE5Model
from pathlib import Path

BASE = Path(__file__).resolve().parent

with open(BASE / "test_cases.json") as f:
    cases = json.load(f)


@pytest.mark.parametrize("case", cases, ids=[f"{i}-{c['name']}" for i, c in enumerate(cases)])
def test_model(case):
    # model = Specter2Model()
    # model = SemanticModel()
    model = MultiLingualE5Model()

    result, _ = model.find_most_similar(
        case["reference"],
        case["other"]
    )
    
    assert result == case["expected"]