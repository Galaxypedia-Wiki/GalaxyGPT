import hashlib
import os
import sys
import traceback
import numpy as np
import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import distances_from_embeddings
from dotenv import load_dotenv
load_dotenv()

GalaxyGPTVersion = "0.2.1"

if not os.getenv("OPENAI_ORG_ID") or not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_ORG_ID and OPENAI_API_KEY in .env")
    sys.exit(1)

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

dataset = str(os.getenv("DATASET")) or "dataset-v3"

################################################################################

print("Loading dataset...")
df = pd.read_csv(os.path.join(__location__, dataset, "embeddings.csv"), index_col=0)
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

df["page_titles"] = pd.read_csv(
    os.path.join(__location__, dataset, "postprocessed.csv"), index_col=0
)["page_title"]


def create_context(question, df, max_len=2000, model="text-embedding-ada-002"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    embeddings = openai.Embedding.create(input=question, engine=model)

    q_embeddings = embeddings["data"][0]["embedding"]

    embeddingsusage = embeddings["usage"]

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values("distances", ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["content"].strip())
        
    print("bingus:", flush=True)
    print(returns, flush=True)
    print('-------------------', flush=True)
    print(df["distances"].values.tolist(), flush=True)

    # Return the context
    return "\n\n###\n\n".join(returns), embeddingsusage


def answer_question(
        df: pd.DataFrame,
        model="gpt-3.5-turbo-0613",
        question="Hello!",
        max_len=2000,
        size="text-embedding-ada-002",
        debug=True,
        max_tokens=250,
        stop_sequence=None,
        datasetver=dataset, 
        username: str | None = None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    
    extrainfo = ""
    # Make sure the question is not empty
    if question == "":
        raise Exception("Question is empty")

    # Make sure the dataframe is not empty
    if df.empty:
        raise Exception("Dataframe is empty")

    # Make sure the question is under 250 tokens
    enc = tiktoken.get_encoding("cl100k_base")
    questiontokens = enc.encode(question)
    if len(questiontokens) > max_tokens:
        raise Exception("Question is too long (max 250 tokens)")

    moderation = openai.Moderation.create(input=question)
    if debug:
        import json

        print(
            "Moderation:\n" + json.dumps(moderation["results"][0], indent=0), flush=True
        )
        print(
            "-----------------------------------------------------------------------------",
            flush=True,
        )

    if moderation["results"][0]["flagged"]:
        raise Exception("Flagged by OpenAI Moderation System")

    context = create_context(question, df, max_len=max_len, model=size)
    print(context)
    embeddingsusage = context[1]
    context = context[0].strip()

    if context == "":
        print("Context is empty", flush=True)
        
        
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context, flush=True)
        print(
            "-----------------------------------------------------------------------------",
            flush=True,
        )

    try:
        # Create a completions using the question and context
        raah = "\n\n"
        response = openai.ChatCompletion.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are GalaxyGPT, a helpful assistant that answers questions about Galaxy, a ROBLOX Space Game.\n"
                               + "The Galaxypedia is the game's official wiki and it owns you\n"
                               + 'The Galaxypedia\'s slogans are "The new era of the Galaxy Wiki" and "A hub for all things Galaxy".\n'
                               + "Answer the question based on the supplied context. If the question cannot be answered, politely say you don't know the answer and ask the user if they have any further questions about Galaxy.\n"
                               + "If the user has a username, it will be provided and you can address them by it. If a username is not provided, do not address the user.\n"
                               + 'Do not reference or mention the "context provided" in your response, no matter what.\n'
                               + 'If a ship infobox is present in the context, prefer using data from within the infobox. An infobox can be found by looking for a wikitext template that has the word "infobox" in its name.\n'
                               + 'If the user is not asking a question (e.g. "thank you", "thanks for the help"): Respond to it and ask the user if they have any further questions.\n'
                               + 'Respond to greetings (e.g. "hi", "hello") with (in this exact order): A greeting, a brief description of yourself, and a question addressed to the user if they have a question or need assistance.\n\n'
                               + 'Steps for responding: First check if the user is asking about a ship (e.g. "what is the deity?", "how much shield does the theia have?"), if so, use the ship\'s wiki page (supplied in the context) and the stats from the ship\'s infobox to answer the question. If you determine the user is not asking about a ship (e.g. "who is <player>?", "what is <item>?"), do your best to answer the question based on the context provided.',
                },
                {
                    "role": "user",
                    "content": f'Context: {context}\n\n---\n\nQuestion: {question}{f"{raah}Username: {str(username)}" if username else ""}',
                    "name": str(username) if username else "",
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
            user=(
                (hashlib.sha256(str(username).encode()).hexdigest())
                if username is not None
                else ""
            ),
        )

        if debug:
            print(
                "User: "
                + (
                    (hashlib.sha256(str(username).encode()).hexdigest())
                    if username is not None
                    else ""
                ),
                flush=True,
            )
            
            return {
                "answer": response["choices"][0]["message"].content.strip(),
                "context": context,
                "tokens": response["usage"],
                "embeddings_usage": embeddingsusage,
                "stop_reason": response["choices"][0]["finish_reason"],
                "dataset": dataset,
                "version": GalaxyGPTVersion,
                "extra": extrainfo.strip()
            }
        else:
            return {
                "answer": response["choices"][0]["message"].content.strip(),
                "dataset": dataset,
                "version": GalaxyGPTVersion,
            }
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        raise e


# print(answer_question(df, question="What are some good strategies to be successful in Galaxy?", debug=True))

if __name__ == "__main__":
    print(
        answer_question(
            df,
            question=sys.argv[1],
            debug=(True if len(sys.argv) < 3 or sys.argv[2] != "False" else False),
        ),
        flush=True,
    )
