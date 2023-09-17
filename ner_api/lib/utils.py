"""Utility Function"""
import math


def chunk_text_for_prediction(text, chunk_size=450):
    # Split the text into words
    words = text.split()

    # Calculate the number of chunks needed
    num_chunks = math.ceil(len(words) / chunk_size)

    # Initialize empty lists for chunked text and corresponding annotations
    chunked_text = []

    # Iterate over the chunks
    for i in range(num_chunks):
        # Determine the start and end indices of the chunk
        start_index = i * chunk_size
        end_index = min(start_index + chunk_size, len(words))

        # Get the words within the chunk
        chunk = words[start_index:end_index]
        chunked_text.append(" ".join(chunk))

    return chunked_text


def align_predicted_annotations(predictions, texts):
    """Aling NER predictions with text"""
    offset = 0
    aligned_preds = []
    for preds, text in zip(predictions, texts):
        for pred in preds:
            pred["start"] += offset
            pred["end"] += offset
            aligned_preds.append(pred)

        # Update offset
        offset = offset + len(text) + 1

    return aligned_preds


def concat_named_entities(model_results, texts):
    """
    Concating named Entities found from model
    """
    results = []
    for model_res, text in zip(model_results, texts):
        res = []
        entity = ""
        score = []
        start = 0
        end = 0
        for item in model_res:
            if item["entity"].startswith("B") or (
                item["entity"].startswith("I") and not item["start"] <= end + 1
            ):
                # If there are records stored resolve them
                if entity != "":
                    res.append(
                        {
                            "entity": entity,
                            "score": sum(score) / len(score),
                            "start": int(start),
                            "end": int(end),
                            "phrase": text[start:end],
                        }
                    )
                # Adding the beginning of new enrtity
                entity = list(item["entity"].split("-"))[1]
                score.append(item["score"])
                start = item["start"]
                end = item["end"]
            elif item["entity"].startswith("I") and item["start"] <= end + 1:
                score.append(item["score"])
                end = item["end"]
        if (
            entity != ""
        ):  # If there are still record at then end of iteration resolve them
            res.append(
                {
                    "entity": entity,
                    "score": sum(score) / len(score),
                    "start": int(start),
                    "end": int(end),
                    "phrase": text[start:end],
                }
            )
        results.append(res)
    return results
