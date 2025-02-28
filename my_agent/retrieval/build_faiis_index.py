import os
import re

import pandas as pd
from markdownify import markdownify as md  # Convert HTML to Markdown format
from openai import OpenAI  # OpenAI API client for generating completions
from tqdm import tqdm  # Progress bar for iterating over DataFrame rows

# Import custom modules for loading Anki data, configuration, and FAISS index management
from my_agent.anki.anki_api import load_anki_query_to_dataframe
from my_agent.config import (
    working_directory_path,
    OPENAI_API_KEY,
    embedding_model,
    anki_query,
)
from my_agent.retrieval.faiss_manager import FAISSManager

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def filter_extras(strings):
    """
    Filters out any strings that contain the word "extra" or "extras" (case-insensitive).

    Args:
        strings (list): List of strings to be filtered.

    Returns:
        list: Filtered list of strings excluding any with "extra"/"extras".
    """
    pattern = (
        r"\bextras?\b"  # Regex pattern to match "extra" or "extras" as whole words
    )
    return [s for s in strings if not re.search(pattern, s, re.IGNORECASE)]


def summarize_text(text, model="gpt-4o-mini"):
    """
    Summarizes the given text into exactly 3 concise lines using the specified model.

    Args:
        text (str): The text to summarize.
        model (str): The model to use for generating the summary.

    Returns:
        str: The generated summary or an empty string if an error occurs.
    """
    try:
        # Define the prompt for summarization
        prompt = f"""
        Summarize the following text in exactly 3 concise lines:

        {text}

        Summary:
        """

        # Request a summary from the OpenAI client (using a chat completion endpoint)
        completion = client.chat.completions.create(
            model=model, store=True, messages=[{"role": "user", "content": prompt}]
        )

        # Return the stripped summary from the response
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return ""

    # This line is unreachable because of the try/except structure above.
    return completion.choices[0].message.content.strip()


def generate_summaries(df, save_every=10, df_summary_path=None):
    """
    Iterates over the DataFrame rows to generate summaries for each row's text.
    Saves the DataFrame to disk every 'save_every' iterations.

    Args:
        df (pd.DataFrame): DataFrame containing the text to summarize.
        save_every (int): Frequency at which the DataFrame is saved.
        df_summary_path (str): File path to save the summarized DataFrame.

    Returns:
        pd.DataFrame: DataFrame updated with a 'summary' column.
    """
    # Work on a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Iterate over each row with a progress bar
    for index, row in tqdm(df.iterrows()):
        df.at[index, "summary"] = summarize_text(row["text"])
        # Save the DataFrame periodically to avoid data loss
        if (index + 1) % save_every == 0:
            df.to_pickle(df_summary_path)
    return df


def filter_extras(strings):
    """
    (Duplicate function) Filters out strings containing "extra" or "extras".

    Args:
        strings (list): List of strings to be filtered.

    Returns:
        list: Filtered list of strings.
    """
    pattern = r"\bextras?\b"  # Regex to match "extra" or "extras"
    return [s for s in strings if not re.search(pattern, s, re.IGNORECASE)]


def prepare_df_for_generate_summaries(df):
    """
    Prepares the DataFrame by merging all Anki fields into a single text field,
    renaming columns, and adding metadata.

    Args:
        df (pd.DataFrame): Original DataFrame loaded from Anki query.

    Returns:
        pd.DataFrame: Processed DataFrame ready for summary generation.
    """
    # Reverse the order of columns and filter out any "extras" fields
    anki_columns = list(df.columns)[::-1]
    anki_columns = filter_extras(anki_columns)  # Remove "extras" fields

    # Combine all selected columns into one string using markdownify for formatting
    df["text"] = df.apply(lambda row: md(row[anki_columns].str.cat()), axis=1)

    # Rename the 'card_number' column to 'id'
    df = df.rename(columns={"card_number": "id"})
    # Initialize an empty metadata column
    df["meta"] = {}

    # Select only the necessary columns for further processing
    df = df[["f", "id", "text", "meta"]]

    return df


if __name__ == "__main__":
    # Initialize the OpenAI client
    client = OpenAI()

    # Build the file path for storing the summarized DataFrame
    df_summary_path = os.path.join(
        working_directory_path, f"artifacts/{anki_query}_summary.pkl"
    )

    # Check if a summarized DataFrame already exists on disk
    if os.path.exists(df_summary_path):
        summarized_df = pd.read_pickle(df_summary_path)
        # Identify rows that have not been summarized yet
        remaining_df = summarized_df[summarized_df.summary.isna()]
        if len(remaining_df) > 0:
            # Prepare the remaining DataFrame and generate summaries for it
            remaining_df = prepare_df_for_generate_summaries(remaining_df)
            remaining_summarized_df = generate_summaries(
                df=remaining_df,
                df_summary_path=df_summary_path.replace("_summary", "_summary_add_on"),
            )
            # Concatenate the already summarized data with the new summaries
            summarized_df = pd.concat(
                [summarized_df[~summarized_df.summary.isna()], remaining_summarized_df]
            ).reset_index(drop=True)
            # Save the updated DataFrame back to disk
            summarized_df.to_pickle(df_summary_path)
    else:
        # If no summarized DataFrame exists, load the Anki data and prepare it
        df = load_anki_query_to_dataframe(anki_query)
        df = prepare_df_for_generate_summaries(df)
        summarized_df = generate_summaries(df=df, df_summary_path=df_summary_path)

    # Convert the summarized DataFrame to a list of dictionaries (documents)
    docs = summarized_df.to_dict(orient="records")

    # Build the FAISS index using the summaries as the index column
    faiss_mgr = FAISSManager(model_name=embedding_model)
    faiss_mgr.build_index(docs, index_col="summary")

    # Save the FAISS index and corresponding metadata to disk
    faiss_mgr.save_index(
        f"{working_directory_path}/artifacts/faiss_{anki_query}.index",
        f"{working_directory_path}/artifacts/metadata_{anki_query}.pkl",
    )
