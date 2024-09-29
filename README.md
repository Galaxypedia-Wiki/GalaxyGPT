## GalaxyGPT
GalaxyGPT is an open-source project that aims to create an AI assistant for Galaxy, a ROBLOX space game. GalaxyGPT works by embedding the Galaxypedia's database, then using cosine similarity to find relevant pages for a user's query. The information is then given to ChatGPT to generate a response.

> [!NOTE]
> This README is a work in progress. At the moment, you're on your own in setting this up. YMMV

> [!NOTE]
> Also note that this project is still in the early stages of development. We recently migrated from python to c# so there's a lot that is missing.

## Technical Details
See [Technical Details](https://blog.smallketchup.ca/galaxypedia/2024/08/14/GalaxyGPT.html#:~:text=in%20your%20pocket!-,technical%20details,-This%20is%20the)

## Datasets & Snapshots
Datasets are database dumps of the Galaxypedia, formatted and stripped away of unnecessary information, providing only what is needed to understand what the contents of a page. Snapshots are already-embedded versions of datasets. These are required for setting up GalaxyGPT. The datasets and snapshots are compressed using the tar.gz format. You should probably be able to decompress them as usual, but in the case you cannot, use a program like [7Zip](https://www.7-zip.org/) to do so. The tar file includes both the dataset and the equivalent snapshot (if possible).

**NOTE:** These datasets are provided under the [CC-BY](https://creativecommons.org/licenses/by/4.0/) license. We ask that you do not use these dumps for any ill intentions, and stay true to the Galaxypedia's mission in your usage. Ensure that you provide appropriate credit to the Galaxypedia.

Latest dataset & snapshot:  
[galaxypedia-2024-09-22.tar.gz](https://github.com/user-attachments/files/17181662/galaxypedia-2024-09-22.tar.gz)


## Setting Up GalaxyGPT
Open the solution in your favorite C# IDE. I recommend JetBrains Rider, but Visual Studio should work as well. You can also use VSCode with the C# and EditorConfig extensions installed.

There is currently no way to run/test the project outside the API, as unit tests are a work in progress.

Oh, and some general requirements:
- [Docker](https://www.docker.com/)
- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download)

### Preparing the database
GalaxyGPT uses Qdrant, a Vector Database. You will have to set this up first before doing anything.

You may use the `Start QDrant standalone (dev)` run configuration to do this, or you can edit the `docker-compose.yaml` and uncomment the ports, then run
```
docker compose up qdrant
```

Now that Qdrant is up, you'll want to keep it running in the background.
### Creating a database
So, you can do two things here. Sometimes, with a new dataset, we'll embed everything for you and package it into a snapshot. This way, you don't have to pay for anything and also don't have to wait for the embeddings to be created. But these are susceptible to breakage, and can be hard to set up on headless servers. So your mileage may vary with these. They're generally the easiest way, however. You can always create a database manually if things don't work out.

#### Snapshots
You can find snapshots bundled with the datasets up in the [Datasets](#Datasets) section.

When you set up Qdrant, the API and web ui are automatically exposed. You can access the web ui at [`http://localhost:6333/dashboard`](http://localhost:6333/dashboard#/collections). You can use this to import the snapshot. You should navigate to the "collections" tab, find the blue button that says "upload snapshot", and select the `.snapshot` file.

Hopefully, if all goes well, you now have a collection called "galaxypedia" with all the embeddings in place. You can now move on to the next step.

#### Manual
To manually create a dataset, you'll want to cd into `dataset-assistant` and run the following:
```
dotnet run -- --help
```
You'll see a help message detailing the possible configuration values. Fill in your OpenAI api key, the path to the dataset, and (if you've changed the port of Qdrant), the url to qdrant.

Here you have two options: Use batching or use sequential embedding.

Batching is the way we'd most recommend if you're not in any kind of hurry. Everything is done on OpenAI's end and its generally just a more seamless process. As an added bonus, you save 50% on the cost of embedding everything. It generally takes around 15 minutes for batching to complete. We'd recommend running it, brewing a cup of coffee, and coming back to check on it. Everything should be done by then. But OpenAI claims that it can take up to 24 hours, so it's a bit of a gamble.

Sequential embedding is what we like to call "doing the embedding one after another". Well, that's a bit of a lie, since we use multithreading to increase throughput, but you get the idea. This is guaranteed to finish within a reasonable amount of time. We send an embedding request for each page and wait for the response, handling the next batch of pages as we go. This is generally more expensive, but it's a lot more reliable. It's also a lot faster, but comes with its own set of problems. For example, if you're on the free tier, or first tier, you can run into ratelimits. Plus, paying more money is never fun.

You pick your poison. If you decide to use batching (which we again, would highly recommend, save your money people!) add `--batch` to the arguments. If not, just run the command without it.

dataset-assistant will automatically take your csv and do a bunch of logic to chunk and embed your dataset, then fill everything into Qdrant. It might take a while, but let it be. We put a lot of work into adding some cool cli progress bars, watch them go!

### Setting things up for the first run
Now that Qdrant is fully populated with the embeddings, you're ready to run GalaxyGPT. If you still have Qdrant running, you can stop it (by pressing ctrl+c or just closing the terminal window).

Development and Production are entirely seperated. If you previously set up Qdrant for development, you'll have to do it again with the production configuration. This is just to make sure that development can't randomly decide to mess with production.

To start, we need to provide some configuration secrets. Create a `.env` file in the same directory as the `docker-compose.yaml`. Fill it in according to the configuration guide below. Come back here when you're done.

## Running
We provide prebuilt docker images for the API based on the latest git commit. You should probably use these for production. You can always build things yourself by passing `--build` to the `docker compose` command.

For production, you can use `docker compose -f docker-compose.yaml -f docker-compose.prod.yaml up -d` to start everything.

For development, just use the classic `docker compose up -d` command.

You can run `docker compose logs -f` to investigate logs and verify everything went as planned. If anything doesn't work as expected, open an issue, discussion, or ask in our discord server, we'd be happy (💀) to help.

The API will be available at `http://localhost:3636`.

To ask a question, send a POST request to `http://localhost:3636/api/v1/ask` with the following JSON body (other things might be needed, i dont know...):
```json
{
  "question": "Your question here"
}
```
The response will be a JSON object with the answer and relevant information.

To figure out all possible endpoints, you can run for development and open the swagger ui at `http://localhost:3636/swagger/index.html`. Or... you can just look at the code. It's not *that* hard to understand.

## Contributing
Contributions are welcome! Please see the [CONTRIBUTING](CONTRIBUTING.md) file for more information.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for more information.

The Galaxypedia itself is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License. Please ensure you are compliant with the license when using the dataset.

In general, I would appreciate that you credit the Galaxypedia if you create derivative works, or if you decide to use the dataset.
