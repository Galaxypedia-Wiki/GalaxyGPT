# Description: This file takes a wikitext dataset with two columns (title and content) and splits the content into chunks of a maximum number of tokens. It then saves the chunks to a csv file for use in other files.

from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

openai.organization = "org-o2qPchMFXjCqioXPTN4pnJot" or os.getenv('OPENAI_ORG')
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.Model.list()

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

###############################################################################
###############################################################################

# Create a list to store the text files
texts=[]

# Create a dataframe from the list of texts
df = pd.read_csv(os.path.join(__location__, 'gpedia.csv'), escapechar="\\", header=0, names=['page_title', 'content'])

# Set the text column to be the raw text with the newlines removed
df['content'] = df.page_title + ". " + df.content
df.to_csv(os.path.join(__location__, 'processed', 'scraped.csv'))
df.head()

################################################################################

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv(os.path.join(__location__, 'processed', 'scraped.csv'), index_col=0)
df.columns = ['title', 'content']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.content.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
#df.n_tokens.hist()

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    

shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['content'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['content'])
    
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['content'] )


###############################################################################

df = pd.DataFrame(shortened, columns = ['content'])
df['n_tokens'] = df.content.apply(lambda x: len(tokenizer.encode(x)))
df.to_csv('processed/tokens or sum lmfao.csv')
df.head()

df['embeddings'] = df.content.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

df.to_csv('processed/embeddings.csv')
df.head()

###############################################################################