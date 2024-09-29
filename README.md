## GalaxyGPT
GalaxyGPT is an open-source project that aims to create an AI assistant for Galaxy. GalaxyGPT works by embedding the Galaxypedia's database, then using cosine similarity to find relevant pages for a user's query. The information is then given to ChatGPT to generate a response.

> [!NOTE]
> This README is a work in progress. At the moment, you're on your own in setting this up. A more comprehensive README & Docs will be provided soon

> [!NOTE]
> Also note that this project is still in the early stages of development. We recently migrated from python to c# so there's a lot that is missing.

## Technical Details
See [Technical Details](https://blog.smallketchup.ca/galaxypedia/2024/08/14/GalaxyGPT.html#:~:text=in%20your%20pocket!-,technical%20details,-This%20is%20the)

## Datasets
Datasets are database dumps of the Galaxypedia, formatted and stripped away of unnecessary information, providing only what is needed to understand what the contents of a page. These are required for setting up GalaxyGPT. The datasets are compressed using GZIP. You should probably be able to decompress them as usual, but in the case you cannot, use a program like [7Zip](https://www.7-zip.org/) to do so.

**NOTE:** These datasets are provided under the [CC-BY](https://creativecommons.org/licenses/by/4.0/) license. We ask that you do not use these dumps for any ill intentions, and stay true to the Galaxypedia's mission in your usage. Ensure that you provide appropriate credit to the Galaxypedia.

- [galaxypedia-2024-09-22.csv.gz](https://github.com/user-attachments/files/17181542/galaxypedia-2024-09-22.csv.gz)

## Setting Up GalaxyGPT
> [!WARNING]
> You cannot currently run GalaxyGPT on your own machine because the Galaxypedia database dumps are not publicly available for the time being. Please check back later about this!

Open the solution in your favorite C# IDE. I recommend JetBrains Rider, but Visual Studio should work as well.

There is currently no way to run the project outside of the API, as unit tests are a work in progress.

The run configurations included within the repo should be helpful in doing the bulk work, you may use those if you wish. But I'll be writing things out manually (the hard way) to be safe.

### Preparing the database
GalaxyGPT uses Qdrant, a Vector Database. You will have to set this up first, before doing anything.

You may use the `Start qdrant standalone` run configuration to do this, or you can edit the `docker-compose.yaml` and uncomment the ports, then run
```
docker compose up qdrant
```

Now that Qdrant is up, you'll want to keep it running in the background.
### Creating a dataset
Cd into `dataset-assistant` and run the following:
```
dotnet run -- --help
```
You'll see a help message detailing the possible configuration values. Fill in your OpenAI api key, the path to the dataset, and (if you've changed the port of Qdrant), the url to qdrant. Then run the resulting command.

dataset-assistant will automatically take your csv and do a bunch of logic to chunk and embed your dataset, then fill everything into Qdrant. It might take a while, but let it be.

### Setting things up for the first run
Now that Qdrant is fully populated with the embeddings, you're ready to run GalaxyGPT. If you still have Qdrant running, you can stop it (by pressing ctrl+c or just closing the terminal window).

Depending on your preference, you may go into `docker-compose.yaml` and comment out the ports again. You don't have to do this, but considering that Qdrant doesn't really have any authentication by default, I would recommend only exposing the ports as necessary. This way, only GalaxyGPT will be able to access Qdrant.

Once you're done with that, create a `.env` file in the same directory as the `docker-compose.yaml`. Fill it in according to the configuration guide below.

Now, run `docker compose up -d` to start everything in the background. Once that command finishes, run `docker compose logs -f` to investigate the logs and verify successful setup. If anything doesn't work as expected, let us know.

## Running
GalaxyGPT is meant to be run with docker.

To run the API, run the `galaxygpt-api` project. The API will be available at `http://localhost:3636`.

To ask a question, send a POST request to `http://localhost:3636/api/v1/ask` with the following JSON body:
```json
{
  "question": "Your question here"
}
```
The response will be a JSON object with the answer and relevant information.

## Contributing
Contributions are welcome! Please see the [CONTRIBUTING](CONTRIBUTING.md) file for more information.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for more information.

The Galaxypedia itself is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License. Please ensure you are compliant with the license when using the dataset.

In general, I would appreciate that you credit the Galaxypedia if you create derivative works, or if you decide to use the dataset.
