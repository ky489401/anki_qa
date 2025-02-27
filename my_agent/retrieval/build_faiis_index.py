import os
import re

import pandas as pd
from markdownify import markdownify as md
from openai import OpenAI
from tqdm import tqdm

from my_agent.anki.anki_api import load_anki_query_to_dataframe
from my_agent.config import working_directory_path, OPENAI_API_KEY, embedding_model
from my_agent.retrieval.faiss_manager import FAISSManager

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def filter_extras(strings):
    pattern = r"\bextras?\b"  # Matches "extra" or "extras" as whole words
    return [s for s in strings if not re.search(pattern, s, re.IGNORECASE)]


def summarize_text(text, model="gpt-4o-mini"):
    try:
        prompt = f"""
        Summarize the following text in exactly 3 concise lines:

        {text}

        Summary:
        """

        completion = client.chat.completions.create(
            model=model, store=True, messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content.strip()
    except Exception as e:
        return ""

    return completion.choices[0].message.content.strip()


def generate_summaries(df, save_every=50, df_summary_path=None):

    df = df.copy()

    for index, row in tqdm(df.iterrows()):
        df.at[index, "summary"] = summarize_text(row["text"])
        if (index + 1) % save_every == 0:
            df.to_pickle(df_summary_path)
    return df


def filter_extras(strings):
    pattern = r"\bextras?\b"  # Matches "extra" or "extras" as whole words
    return [s for s in strings if not re.search(pattern, s, re.IGNORECASE)]


if __name__ == "__main__":
    query = "deck:Quant::ML::Essential -is:suspended"

    df_summary_path = os.path.join(
        working_directory_path, f"my_agent/artifacts/{query}_summary.pkl"
    )

    if os.path.exists(df_summary_path):
        summarized_df = pd.read_pickle(df_summary_path)

    else:
        client = OpenAI()

        df = load_anki_query_to_dataframe(query)

        anki_columns = list(df.columns)[::-1]
        anki_columns = filter_extras(anki_columns)  # remove "extras" fields

        df["text"] = df.apply(
            lambda row: md(row[anki_columns].str.cat()), axis=1
        )  # combine all fields into single field

        df = df.rename(columns={"card_number": "id"})
        df["meta"] = {}

        df = df[["f", "id", "text", "meta"]]

        # Generate 3 line summary for each anki card.
        # Then build index on the 3 line summaries.
        summarized_df = generate_summaries(df=df, df_summary_path=df_summary_path)

    docs = summarized_df.to_dict(orient="records")

    # build index
    faiss_mgr = FAISSManager(model_name=embedding_model)
    faiss_mgr.build_index(docs, index_col="summary")

    # Save index
    faiss_mgr.save_index(
        f"{working_directory_path}/my_agent/artifacts/faiss.index",
        f"{working_directory_path}/my_agent/artifacts/metadata.pkl",
    )
