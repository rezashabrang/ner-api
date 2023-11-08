"""Document processor Endpoint."""
from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import (
    AutoTokenizer,
    TFAutoModelForTokenClassification,
    TokenClassificationPipeline,
)

from ..logger import LOGGER
from ..lib.utils import (
    align_predicted_annotations,
    chunk_text_for_prediction,
    concat_named_entities,
)

# ------------------------------ Initialization -------------------------------
router = APIRouter()

model_path = os.environ["MODEL_PATH"]

TOKENIZER = AutoTokenizer.from_pretrained(model_path)

MODEL = TFAutoModelForTokenClassification.from_pretrained(model_path)

# Defining pipeline
NER_PIPELINE = TokenClassificationPipeline(
    model=MODEL,
    tokenizer=TOKENIZER,
)


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
    """Extract named entities from text."""
    try:
        LOGGER.debug("Chunking text for prediction.")
        # Passing the ner task to pipeline
        ner_results = []
        chunks = chunk_text_for_prediction(doc.text)
        unaligned_preds = []
        for chunk in chunks:
            res = NER_PIPELINE(chunk)
            unaligned_preds.append(res)
        ner_results.append(align_predicted_annotations(unaligned_preds, chunks))

        LOGGER.debug("Received predictions from model.")

        # Curating named entities
        curated_ners = concat_named_entities(
            model_results=ner_results, texts=[doc.text]
        )
        LOGGER.debug(f"Found {len(curated_ners[0])} entities.")
        return {"namedEntities": curated_ners}
    except HTTPException as err:
        raise HTTPException(status_code=400) from err

    except Exception as err:
        raise HTTPException(status_code=400) from err
