"""Document processor Endpoint."""
import re
from typing import Dict, Optional

from numpy import float32

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, TokenClassificationPipeline
from pathlib import Path


def convert_types(result: list):
    res = []
    for item in result:
        temp_res = {}
        for _, (key, value) in enumerate(item.items()):
            if type(value) == float32:
                value = float(value)
            temp_res[key] = value

        res.append([temp_res])
    return res

def concat_named_entities(model_res, text):
    """
    Concating named Entities found from model
    """
    res = []
    entity = ""
    score = []
    start = 0
    end = 0
    for item in model_res:
        if item["entity"].startswith("B"):
            # If there are records stored resolve them
            if entity != "":      
                res.append(
                    {
                        "entity": entity,
                        "score": sum(score) / len(score),
                        "start": start,
                        "end": end,
                        "word": text[start:end]
                    }
                )
            # Adding the beginning of new enrtity
            entity = list(item["entity"].split("-"))[1]
            score.append(item["score"])
            start = item["start"]
            end = item["end"]
        elif item["entity"].startswith("I"):
            score.append(item["score"])
            end = item["end"]
    if entity != "":  # If there are still record at then end of iteration resolve them
        res.append(
                    {
                        "entity": entity,
                        "score": sum(score) / len(score),
                        "start": start,
                        "end": end,
                        "word": text[start:end]
                    }
                )
    return res






# ------------------------------ Initialization -------------------------------
router = APIRouter()

model_path = os.environ["MODEL_PATH"]

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = TFAutoModelForTokenClassification.from_pretrained(model_path)

# ---------------------------- function definition ----------------------------


class TextData(BaseModel):
    """Schema for comments"""
    text: str


@router.post(
    "/api/ner/",
    response_model=dict,
    tags=["NER"],
    status_code=200,
)
async def extract_named_entities(
    doc: TextData,
) -> Dict[str, str]:
    """
    """
    try:
        pipe = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
        res = pipe([doc.text])
        res = concat_named_entities(res[0], doc.text)
        return {"namedEntities":res}
    except HTTPException as err:
        raise HTTPException(status_code=400) from err

    except Exception as err:
        raise HTTPException(status_code=400) from err
