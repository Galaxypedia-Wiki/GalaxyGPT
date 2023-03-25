import dotenv from 'dotenv'
dotenv.config()

// Load modules
import openaimodule from 'openai'
import tiktoken from '@dqbd/tiktoken'
import fs from 'fs'
import { json2csv } from 'json-2-csv'
import {generate, parse, transform, stringify} from 'csv';

import { fileURLToPath } from 'url';
import path from 'path'
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import express from 'express'
const app = express()
const tokenizer = tiktoken.encoding_for_model('text-embedding-ada-002')
const configuration = new openaimodule.Configuration({
	apiKey: process.env.OPENAI_API_KEY,
})
const openai = new openaimodule.OpenAIApi(configuration)

// Config
const dataset_path = './src/datasets/gpedia.csv'

// Load the dataset into memory
var csv = [];
await new Promise((resolve, reject) => {
	fs.createReadStream(dataset_path)
	.pipe(parse({delimiter: ',', quote: '"', escape: '\\', columns: true, trim: true}))
	.on('data', async (row) => {
		csv.push(row)
	})
	.on('end', () => {
		resolve()
	})
})

// Tokenize the dataset and add the number of tokens to a new column
async function tokenizeDataset() {
	for (const [index, data] of Object.entries(csv)) {
		csv[index].tokens = tokenizer.encode(data.content).length
	}
}
var max_tokens = 500

// Function to split the text into chunks of a maximum number of tokens
function splitText(text, max_tokens) {
	const sentences = text.split('. ')

	// Number of tokens for each sentence
	const tokens = sentences.map(sentence => tokenizer.encode(" " + sentence).length)

	var chunks = []
	var tokens_so_far = 0
	var chunk = []

	// Loop through the sentences and tokens joined together in a tuple
	for (const [index, sentence] of Object.entries(sentences)) {
		// If the number of tokens so far plus the number of tokens in the current sentence is greater 
        // than the max number of tokens, then add the chunk to the list of chunks and reset
        // the chunk and tokens so far

		const token = tokens[index]

		if (tokens_so_far + token > max_tokens) {
			chunks.push(chunk.join('. '))
			chunk = []
			tokens_so_far = 0
		}

		if (token > max_tokens) {
			continue
		}

		chunk.push(sentence)
		tokens_so_far += token + 1
	}

	return chunks
}

var shortened = []

async function chunkShit() {
// Iterate through the dataset and split the text into chunks of a maximum number of tokens, row by row
for (const [index, data] of Object.entries(csv)) {
	if (!data.content) continue

	// If the number of tokens is greater than the max number of tokens, split the text into chunks
	if (data.tokens > max_tokens) {
		shortened.push(splitText(data.content, max_tokens))
	} else {
		shortened.push(data.content)
	}

}
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// Create the embeddings for the dataset
async function createEmbeddings() {
const boogas = shortened.map(async (text) => {
		var x = await openai.createEmbedding({ input: text, model: 'text-embedding-ada-002' })
		await sleep(1000)
		return x.data.data[0].embedding
	})
	csv.embeddings = await Promise.all(boogas)
}

// Write the dataset to a csv file
async function writeDataset() {
	const actualcsv = await json2csv(csv)
	fs.writeFileSync('./gpediaembeddings.csv', actualcsv)
}


app.use(express.static(__dirname + '/ui'))

app.listen(3636, () => {
	console.log(`we up`)
  })

app.post('/api/v1/ask', async (req, res) => {
	const pagename = req.query.pagename
	const prompt = req.query.prompt

	if (!pagename || !prompt) {
		res.status(500)
		res.send('Missing parameters')
		return
	}

	const datasearch = csv.find((row) => row.page_title.toLowerCase() === pagename.toLowerCase())
	var data = ""

	if (!datasearch) {
		//throw new Error("Page not found")
		return res.status(500).send("Page not found")
	} else {
		data = datasearch.content
	}

	console.log(data)
	var response = null

	try {
		response = await openai.createChatCompletion({
			model: 'gpt-3.5-turbo',
			messages: [
				{ role: 'system', content: '\
				You are a helpful assistant that answers questions regarding the Galaxypedia, a site based upon mediawiki.\
				The site\'s slogan is "The new era of the Galaxy Wiki".\
				Refer to yourself as "GalaxyGPT"\
				' },
				{ role: 'user', content: '\
				Read the data supplied and try to answer the prompt based upon the supplied data, if uncertain, reply with a message notifying the user that you failed to answer their question. Do not refer to "data"\n\n\
				Prompt: ' + prompt + '\n\nData: ' + data},
			],
			temperature: 0
		})
		console.log(JSON.stringify(response.data, null, 2))
	} catch (err) {
		console.error(JSON.stringify(err, null, 2))
		res.status(500)
		res.send(err.message)
		return
	}

	res.send(response.data.choices[0].message)
})

/* await tokenizeDataset()
await chunkShit()
await createEmbeddings()
await writeDataset()
 */

console.log('finished')