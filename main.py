# Entrypoint for GalaxyGPT

## Initalization
import hashlib
import os
import subprocess
import threading
import time
import traceback
import warnings

import colorama
import numpy as np
import pandas as pd
import schedule
import tiktoken
from discord_webhook import DiscordWebhook
from dotenv import load_dotenv
from openai import OpenAI
from scipy.spatial.distance import cosine

load_dotenv()

GalaxyGPTVersion = os.getenv("VERSION")
if GalaxyGPTVersion is None:
    raise Exception("Please set VERSION in .env")

if not os.getenv("OPENAI_ORG_ID") or not os.getenv("OPENAI_API_KEY"):
    raise Exception("Please set OPENAI_ORG_ID and OPENAI_API_KEY in .env")

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

dataset = str(os.getenv("DATASET")) # Dataset to use
if dataset is None:
    raise Exception("Please set DATASET in .env")
context_len = int(os.getenv("MAX_CONTEXT_LEN")) # Maximum context length in tokens
if context_len is None:
    raise Exception("Please set MAX_CONTEXT_LEN in .env")
gpt_model = str(os.getenv("MODEL")) # Model to use
if gpt_model is None:
    raise Exception("Please set MODEL in .env")

print("GalaxyGPT v" + GalaxyGPTVersion + " - " + dataset + " - " + str(context_len) + " max len")

################################################################################
# Load datasets

def loadDataset():
    print("Loading dataset...")
    global df
    
    df = pd.read_csv(os.path.join(__location__, dataset, "embeddings.csv"), index_col=0)
    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

    df["page_titles"] = pd.read_csv(os.path.join(__location__, dataset, "processed.csv"), index_col=0)["page_title"]
    print("Dataset loaded!")

loadDataset()

################################################################################
# Functions

def strtobool(string: str):
    """
    Returns a boolean value based on the given string.
    True values are 'y', 'yes', 't', 'true', 'on', and '1',
    False values are 'n', 'no', 'f', 'false', 'off', and '0'.  
    Raises ValueError if 'string' is anything else.
    """
    if string.lower() in ["y", "yes", "t", "true", "on", "1"]: 
        return True
    elif string.lower() in ["n", "no", "f", "false", "off", "0"]: 
        return False
    else: 
        raise ValueError(f"invalid string given: {string}")

def create_context(question, df, max_len=context_len, model="text-embedding-ada-002", debug=True):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    embeddings = openai_client.embeddings.create(input=question, model=model)

    q_embeddings = embeddings.data[0].embedding

    embeddingsusage = embeddings.usage
    # Get the distances from the embeddings
    df["distances"] = df["embeddings"].apply(lambda x: cosine(q_embeddings, x))

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
        
    if debug:
        print("bingus:", flush=True)
        print(returns, flush=True)
        print("-------------------", flush=True)
        print(df["distances"].values.tolist(), flush=True)

    # Return the context
    return "\n\n###\n\n".join(returns), embeddingsusage


def answer_question(
        df: pd.DataFrame,
        model=gpt_model,
        question="Hello!",
        max_len=context_len,
        size="text-embedding-ada-002",
        debug=True,
        max_tokens=250,
        stop_sequence=None,
        username: str | None = None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    
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
        raise Exception("Question is too long")

    moderation = openai_client.moderations.create(input=question)

    import json
    
    if debug:
        print(
            "Moderation:\n" + str(moderation.results[0]), flush=True
        )
        print(
            "-----------------------------------------------------------------------------",
            flush=True,
        )

    if moderation.results[0].flagged:
        raise Exception("Flagged by OpenAI Moderation System")

    context = create_context(question, df, max_len=max_len, model=size, debug=debug)
    embeddingsusage = context[1]
    context = context[0].strip()

    if context == "":
        warnings.warn("Context is empty")
        
        
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
        response = openai_client.chat.completions.create(
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
                               + 'Steps for responding:\nFirst check if the user is asking about a ship (e.g. "what is the deity?", "how much shield does the theia have?"), if so, use the ship\'s wiki page (supplied in the context) and the stats from the ship\'s infobox to answer the question. If you determine the user is not asking about a ship (e.g. "who is <player>?", "what is <item>?"), do your best to answer the question based on the context provided.',
                },
                {
                    "role": "user",
                    "content": f'Context: {context}\n\n---\n\nQuestion: {question}{f"{raah}Username: {str(username)}" if username else ""}',
                    "name": str(username) if username else "None",
                },
            ],
            temperature=0.2,
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

        response = json.loads(response.model_dump_json())

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
                "answer": response["choices"][0]["message"]["content"].strip(),
                "context": context,
                "tokens": response["usage"],
                "embeddings_usage": {"prompt_tokens": embeddingsusage.prompt_tokens, "total_tokens": embeddingsusage.total_tokens},
                "stop_reason": response["choices"][0]["finish_reason"],
                "dataset": dataset,
                "version": GalaxyGPTVersion,
                "model": model
            }
        else:
            return {
                "answer": response["choices"][0]["message"]["content"].strip(),
                "dataset": dataset,
                "version": GalaxyGPTVersion,
            }
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        raise e

# Automatic Dataset Creation System
class ADCS:
    timer = None
    timerbreak = False
    status = "Stopped" # Acceptable Values: "Stopped", "Running"
    webhook = DiscordWebhook(url=os.getenv("DISCORD_WEBHOOK_URL"))
    
    @staticmethod
    def reloadDataset(newdataset):
        global df, dataset
        
        if not os.path.exists(os.path.join(__location__, newdataset, "embeddings.csv")):
            raise Exception("Embeddings were not generated")
        
        try:
            del df
            dataset = newdataset
            loadDataset()
        except Exception as e:
            print(traceback.format_exc(), flush=True)
            raise e
    
    @staticmethod
    def createDataset(reload=False, webhook=None, noembeddings=False):
        global df, dataset
        if os.getenv("DATABASE_PASSWORD") is None:
            raise Exception("Please set DATABASE_PASSWORD in .env")
        
        # Generate the dataset
        print(colorama.Fore.CYAN + "ADCS:" + colorama.Fore.RESET + " Generating a new dataset...")
        subprocess.run(["./generate-dataset.sh", os.getenv("DATABASE_PASSWORD")], cwd=os.path.join(__location__))
        
        print(colorama.Fore.CYAN + "ADCS:" + colorama.Fore.RESET + " Preparing the dataset...")
        # Prepare the dataset
        if noembeddings is True:
            print("Won't be generating embeddings for this dataset")
            subprocess.run(["python3", "dataset.py", "-o", "dataset-ADCS", "--max-len", str(int(context_len/2)), "--no-embeddings", "--cleandir"], cwd=os.path.join(__location__))
        else:
            subprocess.run(["python3", "dataset.py", "-o", "dataset-ADCS", "--max-len", str(int(context_len/2)), "--cleandir"], cwd=os.path.join(__location__))

        if reload is True and noembeddings is False:
            ADCS.reloadDataset("dataset-ADCS")
            
        print("Dataset created!")
        
        if ADCS.webhook:
            ADCS.webhook.set_content("Created new dataset")
            ADCS.webhook.execute()
    
    @staticmethod
    def start():
        if ADCS.status == "Running":
            print(colorama.Fore.CYAN + "ADCS:" + colorama.Fore.RESET + " Already running!")
            return
        
        print(colorama.Fore.CYAN + "ADCS:" + colorama.Fore.RESET + " Starting scheduler to run at 00:00...")
        ADCS.timer = schedule.every().day.at("00:00").do(ADCS.createDataset, True)
        
        def loop():
            while ADCS.timerbreak is False:
                schedule.run_pending()
                time.sleep(1)
            print(colorama.Fore.CYAN + "ADCS:" + colorama.Fore.RESET + " Stopped!")
                
        threading.Thread(target=loop).start()
        print(colorama.Fore.CYAN + "ADCS:" + colorama.Fore.RESET + " Started!")
        ADCS.status = "Running"
        
        if ADCS.webhook:
            ADCS.webhook.set_content("ADCS started")
            ADCS.webhook.execute()
    
    @staticmethod
    def stop():
        if ADCS.status == "Stopped":
            print(colorama.Fore.CYAN + "ADCS:" + colorama.Fore.RESET + " Already stopped!")
            return
        print(colorama.Fore.CYAN + "ADCS:" + colorama.Fore.RESET + " Stopping...")
        ADCS.timerbreak = True
        ADCS.status = "Stopped"
        
        if ADCS.webhook:
            ADCS.webhook.set_content("ADCS stopped")
            ADCS.webhook.execute()

adcsdefault = strtobool(os.getenv("ADCS", "False"))
if adcsdefault is True:
        scheduler = ADCS()
        if scheduler.status == "Stopped":
            scheduler.start()
elif adcsdefault is False:
    print("ADCS is currently disabled")            

if __name__ == "__main__":
    print("Type exit or quit to exit")
    while True:
        question = str(input("\n" + "\x1B[4m" + "User" + "\x1B[0m" + "\n"))
        if not question or question.lower() == "exit" or question.lower() == "quit":
            break
        
        question = question.strip()
        response = answer_question(df, question=question, debug=False)
        print("\n" + "\x1B[4m" + "GalaxyGPT" + "\x1B[0m" + "\n" + response["answer"])