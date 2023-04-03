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

from dotenv import load_dotenv
load_dotenv()


openai.organization = "org-o2qPchMFXjCqioXPTN4pnJot"
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.Model.list()

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

max_tokens = 500

################################################################################

df=pd.read_csv(os.path.join(__location__, 'processed', 'embeddings.csv'), index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()

def create_context(
    question, df, max_len=1800, size="ada"
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
    max_len=1800,
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
                { "role": "system", "content": '\
				You are a helpful assistant that answers questions about Galaxy, a Sci-Fi ROBLOX Space Game. You can assume most of your questions will be regarding ships and their stats\
				The site\'s slogan is "The new era of the Galaxy Wiki".\
				Refer to yourself as "GalaxyGPT"\
                You were trained using data gathed from the Galaxypedia\
                You are the property of the Galaxypedia\
				' },
				{ "role": 'user', "content": f'\
				Answer the question based on the supplied context. If uncertain, reply with a message notifying the user that you failed to answer their question. Do not refer to "data". If a ship infobox is present, prefer using the data from within the infobox\n\n\
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
        return response['choices'][0]['message'].content.strip()
    except Exception as e:
        print(e)
        return ""
    
#print(answer_question(df, question="What are some good strategies to be successful in Galaxy?", debug=True))

if __name__ == "__main__":
    print(answer_question(df, question=sys.argv[1], debug=(True if len(sys.argv) < 3 or sys.argv[2] != "False" else False)))
