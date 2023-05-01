from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import matplotlib.pyplot as plt
import sys
import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv
load_dotenv()


openai.organization = os.getenv('OPENAI_ORG_ID')
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.Model.list()

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

max_tokens = 500

################################################################################

df=pd.read_csv(os.path.join(__location__, 'processed', 'embeddings.csv'), index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

idfk = pd.read_csv(os.path.join(__location__, 'processed', 'scraped.csv'), index_col=0)
print(idfk['page_title'])

""" chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-ada-002"
            )
collection = chroma_client.get_or_create_collection(name="gpedia", embedding_function=openai_ef)
collection.add(
    documents=df["content"],
    embeddings=df["embeddings"],
    ids=idfk["page_title"],
)

print(collection.peek())

def queryorsumidk(what):
    collection.query(
        query_texts=[what],
    ) """

def create_context(
    question, df, max_len=2000, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["content"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="Hello!",
    max_len=2000,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
            messages=[
                {
                    "role": "system",
                    "content": '\
                    You are GalaxyGPT, a helpful assistant that answers questions about Galaxy, a ROBLOX Space Game.\
                    Answer as concisely as possible.\
                    The Galaxypedia is the game\'s wiki\
                    The Galaxypedia\'s slogan is "The new era of the Galaxy Wiki".\
                    You were trained using data gathered from the Galaxypedia\
                    You are the property of the Galaxypedia\
                    Respond to greetings (e.g. "Hi", "Hello") with (in order) a greeting, a brief introduction and description of yourself, and asking the user if they have a question or need assistance.\
				' },
				{
                    "role": 'user',
                    "content": f'\
                    Answer the question based on the supplied context. If uncertain, reply with a message notifying the user that you failed to answer their question. Do not refer to "data". If a ship infobox is present in the context, prefer using the data from within the infobox. An infobox can be found by looking for wikitext template(s) that has the word "infobox" in its name. Some steps: First check if the user is asking about a ship (e.g. "What is the Deity?", "How much shield does the theia have?"), if so, use the ship\'s wiki page (supplied in the context) and the stats from the ship\'s infobox to answer the question. If you determine the user is not asking about a ship, do your best to answer the question with the context provided.\n\n\
                    Prompt: {question}\n\nContext: {context}' },
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,    
        )
        
        if debug:
            return {
                "answer": response['choices'][0]['message'].content.strip(),
                "context:": context,
                "tokens": response['usage'],
                "stop_reason": response['choices'][0]['finish_reason'],
            }
        else:
            return {
                "answer": response['choices'][0]['message'].content.strip()
            }
    except Exception as e:
        print(e)
        return ""
    
#print(answer_question(df, question="What are some good strategies to be successful in Galaxy?", debug=True))

if __name__ == "__main__":
    print(answer_question(df, question=sys.argv[1], debug=(True if len(sys.argv) < 3 or sys.argv[2] != "False" else False)))
