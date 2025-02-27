import re

from markdownify import markdownify as md

from anki_api import load_anki_query_to_dataframe
from retrieval.faiss_manager import FAISSManager


def filter_extras(strings):
    pattern = r"\bextras?\b"  # Matches "extra" or "extras" as whole words
    return [s for s in strings if not re.search(pattern, s, re.IGNORECASE)]


if __name__ == "__main__":

    df = load_anki_query_to_dataframe('"deck:Quant::ML::Essential::7. GenAI"')

    anki_columns = list(df.columns)[::-1]
    anki_columns = filter_extras(anki_columns)  # remove "extras" fields

    df["text"] = df.apply(lambda row: md(row[anki_columns].str.cat()), axis=1)

    df = df.rename(columns={"card_number": "id"})
    df["meta"] = {}

    df = df[["id", "text", "meta"]]

    docs = df.to_dict(orient="records")

    faiss_mgr = FAISSManager()
    faiss_mgr.build_index(docs)

    # Save index
    faiss_mgr.save_index("artifacts/faiss.index", "artifacts/metadata.pkl")
