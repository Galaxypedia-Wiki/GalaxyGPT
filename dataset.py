# Dataset Preparation for GalaxyGPT

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import time
from glob import glob

import colorama
import pandas as pd
import tiktoken
from colorama import Fore
from dotenv import load_dotenv
from halo import Halo
from openai import OpenAI
from tqdm import tqdm

colorama.init()
tqdm.pandas()
load_dotenv()

olddatasets = [f for f in os.listdir(".") if re.match(r"^dataset-v\d$", f, flags=re.MULTILINE | re.IGNORECASE)]
 
# Arguments
parser = argparse.ArgumentParser("GalaxyGPT Dataset Assistant", description="Generate a dataset for use with GalaxyGPT")
parser.add_argument("--outdir", "-o", help="The output directory", type=str, required=True)
parser.add_argument("--cleandir", help="Delete the contents of the output directory if it exists", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--no-embeddings", help="Don't generate embeddings", action="store_true")
parser.add_argument("--api-key", help="The OpenAI API key to use (defaults to env.OPENAI_API_KEY)", type=str, default=os.getenv("OPENAI_API_KEY"))
parser.add_argument("--org-id", help="The OpenAI organization ID to use (defaults to env.OPENAI_ORG_ID)", type=str, default=os.getenv("OPENAI_ORG_ID"))
parser.add_argument("--dump-database", help="Generate a new database dump for use with this script", action="store_true", default=False)
parser.add_argument("--max-len", help="The maximum token length of a chunk (HIGHLY ADVISED TO SET THIS AS THE (MAXIMUM CONTEXT LIMIT / 2))", type=int, required=True)
parser.add_argument("--compress-old-datasets", help="Compress old datasets into their own respective tar.gz files so long as they follow the dataset-vX naming scheme", action="store_true", default=False)
parser.add_argument("--dataset", help="The path to the datset to use. Defaults to galaxypedia-*.csv, where * is the highest number found in the directory, or to galaxypedia.csv", type=pathlib.Path, default=None, nargs="?")
args = parser.parse_args()

# Get list of old datasets and compress them
if args.compress_old_datasets and len(olddatasets) != 0:
    print("Compressing old datasets...")
    for f in olddatasets:
        spin = Halo(text=f"Compressing {f}...", spinner="dots")
        spin.start()
        subprocess.run(["tar", "-czf", f"{f}.tar.gz", f], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        spin.succeed(f"Compressed {f}!")
    print("Done!")
    deleteq = input("Would you like to delete the old datasets? (Y/n): ")
    if deleteq == "y" or deleteq == "":
        for f in olddatasets:
            spin = Halo(text=f"Deleting {f}...", spinner="dots")
            spin.start()
            shutil.rmtree(f)
            spin.succeed(f"Deleted {f}!")

if args.api_key is None:
    raise Exception("No OpenAI API key specified!")
if args.org_id is None:
    raise Exception("No OpenAI organization ID specified!")
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)
embeddings_model = str(os.getenv("EMBEDDING_MODEL")) # Embeddings Model to use
if embeddings_model is None:
    embeddings_model = "text-embedding-3-small"

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


outdir: str = args.outdir

if outdir == "" or outdir is None:
    raise Exception("No output directory specified!")
print(Fore.GREEN + "Saving results to " + outdir + "!")

if not os.path.exists(outdir):
    os.makedirs(outdir)
    
# Check if there are any files in the output directory
if os.listdir(outdir):
    if args.cleandir is None:
        os.system("cls" if os.name == "nt" else "clear")
        clear_existing_files = input(f"{Fore.YELLOW}{outdir} contains existing files!{Fore.RESET}\nWould you like to delete the contents of {outdir}? (Y/n): ")
            
        if str(clear_existing_files).strip() == "y" or str(clear_existing_files).strip() == "":
            shutil.rmtree(outdir)
            os.makedirs(outdir)
            print("Deleted the contents of " + outdir + "!")
            os.system("cls" if os.name == "nt" else "clear")
    elif args.cleandir:
        shutil.rmtree(outdir)
        os.makedirs(outdir)
        print("Deleted the contents of " + outdir + "!")
        
        
# Load the dataset as a dataframe
if not args.dump_database:
    if args.dataset is None:
        # get the first file that matches the glob, prioritizing the file with the largest number by sorting the list numerically then reversing the list
        pathlist = sorted(glob(__location__ + "/galaxypedia*.csv"), reverse=True)
        if pathlist == []:
            raise Exception("Dataset starting with \'galaxypedia\' and ending with \'.csv\' could not be found!")
        datasetpath: str = pathlist[0]
    else:
        if not str(args.dataset).endswith(".csv"):
            raise Exception("Dataset must be a csv file!")
        if not os.path.exists(args.dataset):
            raise Exception("Dataset does not exist!")
        datasetpath: str = args.dataset
            

if args.dump_database:
    # Rename the old galaxypedia.csv to galaxypedia.csv.old by searching for anything matching galaxypedia*.csv
    for file in glob(__location__ + "/galaxypedia*.csv"):
        print("Renaming galaxypedia*.csv to galaxypedia*.csv.old")
        os.rename(os.path.join(file), os.path.join(file + ".old"))

    print("Generating dataset...")
    try:
        subprocess.run(["/bin/bash", __location__ + "/dump-database.sh"], cwd=__location__, capture_output=True, check=True)
    except Exception as e:
        raise Exception("Failed to generate dataset! " + str(e))
    print("Generated dataset!")

    pathlist = sorted(glob(__location__ + "/galaxypedia*.csv"), reverse=True)
    datasetpath: str = pathlist[0]

# If args.dataset is an absolute path, get the filename
datasetname = os.path.basename(datasetpath)

# if args.dataset is a relative path, get the absolute path
if not os.path.isabs(datasetpath):
    datasetpath = os.path.join(__location__, datasetpath)
 
###############################################################################
###############################################################################


def remove_newlines(serie):
    serie = serie.str.replace("\n", " ")
    serie = serie.str.replace("\\n", " ")
    serie = serie.str.replace("  ", " ")
    serie = serie.str.replace("  ", " ")
    return serie

spinner = Halo(text=f"Loading {str(datasetname)}", spinner="dots")
spinner.start()
df = pd.read_csv(
    datasetpath,
    escapechar="\\",
    header=0,
    names=["page_title", "content"],
)

# grab the page titles for later
page_titles = df.page_title.str.lower().str.replace("_", " ").str.strip()

spinner.succeed(f"Loaded {str(datasetname)}!")


# Sanitize the dataset's contents to make it more readable for the model
spinner = Halo(text="Sanitizing dataset", spinner="dots")
# Remove newlines
contentprocessed = remove_newlines(df.content)
# Remove Gallery tags
galleryregex = re.compile(r"(\|image.?=.?)?<gallery.*?>.*?<\/gallery>\\?\n?", re.S)
contentprocessed = contentprocessed.str.replace(
    galleryregex,
    "", regex=True,
)
# Remove links to files
spinner.text = "Sanitizing dataset (removing links to files)"
fileregex = re.compile(r"\[\[File:.*?\]\]\\?", re.S)
contentprocessed = contentprocessed.str.replace(fileregex, "", regex=True)

# Remove magic words
spinner.text = "Sanitizing dataset (removing magic words)"
magicregex = re.compile(r"__.*?__", re.S)
contentprocessed = contentprocessed.str.replace(magicregex, "", regex=True)

# Remove HTML comments (<!-- -->)
spinner.text = "Sanitizing dataset (removing HTML comments)"
commentregex = re.compile(r"<!--.*?-->\\?\n?", re.S)
contentprocessed = contentprocessed.str.replace(commentregex, "", regex=True)

# Remove span and br tags
spinner.text = "Sanitizing dataset (removing span and br tags)"
spanregex = re.compile(r"<span.*?>|<\/span>\\?\n?|<br.*?>\\?\n?", re.S)
contentprocessed = contentprocessed.str.replace(spanregex, "", regex=True)

# Remove div tags
spinner.text = "Sanitizing dataset (removing div tags)"
divregex = re.compile(r"<div.*?>|<\/div>\\?\n?", re.S)
contentprocessed = contentprocessed.str.replace(divregex, "", regex=True)
spinner.succeed()

spinner = Halo(text="Saving sanitized dataset", spinner="dots")

# Remove rows with empty content
df["content"] = contentprocessed.str.strip()
rows_to_drop = df[df["content"]==""].index
df.drop(rows_to_drop, inplace=True)
# Assemble the final page content
df["content"] = df.page_title.str.lower().str.replace("_", " ").str.strip() + ". " + df.content.str.strip()
# Save
df.to_csv(os.path.join(__location__, outdir, "processed.csv"))
spinner.succeed("Saved sanitized dataset!")

################################################################################

# Load the cl100k_base tokenizer which is designed to work with the text-embedding-ada-002 and text-embedding-3-small models
tokenizer = tiktoken.get_encoding("cl100k_base")

spinner = Halo(text="Loading sanitized dataset", spinner="dots")
df = pd.read_csv(os.path.join(__location__, outdir, "processed.csv"), index_col=0)
df.columns = ["page_title", "content"]
spinner.succeed("Loaded sanitized dataset!")

# Tokenize the text and save the number of tokens to a new column
tqdm.pandas(desc="Tokenizing", leave=False)
df["n_tokens"] = df.content.progress_apply(lambda x: len(tokenizer.encode(x)))
print(Fore.GREEN + "✔ " + Fore.RESET + "Tokenized!")

# Max tokens per chunk
max_tokens: int = args.max_len

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences: str = text.split(". ")

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

# in embeddings_pages_by_row, each index corresponds to a page title.
# for each iteration, append a number to embeddings_pages_by_row
# that is the amount of pages in that index
embeddings_pages_by_row = []

# Loop through the dataframe
itrows = tqdm(df.iterrows(), total=df.shape[0], desc="Splitting dataset into chunks", leave=False)

for row in itrows:
    # If the text is None, go to the next row
    if row[1]["content"] is None:
        embeddings_pages_by_row.append(0) # there is 0 pages for this pagetitle index
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks and
    # add them to the list of shortened texts. Append the number of chunks to the embeddings pages by row
    if row[1]["n_tokens"] > max_tokens:
        chunks = split_into_many(row[1]["content"])
        shortened += chunks
        embeddings_pages_by_row.append(chunks.__len__())  # there is chunks.len pages for this pagetitle index

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]["content"])
        embeddings_pages_by_row.append(1) # there is 1 page for this pagetitle index

print(Fore.GREEN + "✔ " + Fore.RESET + "Dataset split into chunks!")

###############################################################################

df = pd.DataFrame(shortened, columns=["content"])

tqdm.pandas(desc="Tokenizing", leave=False)

# make a list of the number of embeddings rows for each page
embedding_page_titles = []
for i, repeats in enumerate(embeddings_pages_by_row):
    for j in list(range(repeats)):
        embedding_page_titles.append(page_titles[i + 1]) # +1 because page_titles[0] is the column name, not the first entry

df["page_title"] = embedding_page_titles

df["n_tokens"] = df.content.progress_apply(lambda x: len(tokenizer.encode(x)))
print(Fore.GREEN + "✔ " + Fore.RESET + "Tokenized!")

spinner = Halo(text="Saving tokenized dataset", spinner="dots")
df.to_csv(os.path.join(__location__, outdir, "tokenized.csv"))
spinner.succeed("Saved tokenized dataset!")

if args.no_embeddings is False:
    cost = 0
    
    baller = tqdm(total=df.shape[0], desc="Embedding", leave=False)
    def idk(x):
        global cost
        cost += (len(tokenizer.encode(x)) / 1000) * 0.0001
        baller.set_postfix_str(str(round(cost, 8)))
        baller.update(1)

        return openai_client.embeddings.create(input=x, model=embeddings_model).data[0].embedding
    
    df["embeddings"] = df.content.apply(idk)
    baller.close()
    print(Fore.GREEN + "✔ " + Fore.RESET + "Embedded!")

    spinner = Halo(text="Saving embedded dataset", spinner="dots")
    df.to_csv(os.path.join(__location__, outdir, "embeddings.csv"))
    spinner.succeed("Saved embedded dataset!")

spinner = Halo(text="Copying initial dataset to output directory", spinner="dots")
shutil.copyfile(datasetpath, os.path.join(__location__, outdir, datasetname))
spinner.succeed("Copied initial dataset to output directory!")

with open(os.path.join(__location__, outdir, "METADATA.txt"), "w") as file:
    file.write(f"Dataset: {datasetname}\nTimestamp: {time.ctime(time.time())}\nMax_len: {max_tokens}")

print("Done!")
