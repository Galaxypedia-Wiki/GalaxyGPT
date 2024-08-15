## GalaxyGPT
GalaxyGPT is an open-source project that aims to create an AI assistant for Galaxy. GalaxyGPT works by embedding the Galaxypedia's database, then using cosine similarity to find relevant pages for a user's query. The information is then given to ChatGPT to generate a response.

> [!NOTE]
> This README is a work in progress. Please check back later for more information.

> [!NOTE]
> Also note that this project is still in the early stages of development. We recently migrated from python to c# so there's a lot that is missing.

## Technical Details
See [Technical Details](https://blog.smallketchup.ca/galaxypedia/2024/08/14/GalaxyGPT.html#:~:text=in%20your%20pocket!-,technical%20details,-This%20is%20the)

## Setting Up GalaxyGPT
> [!WARNING]
> You cannot currently run GalaxyGPT on your own machine because the Galaxypedia database dumps are not publicly available for the time being. Please check back later about this!

Open the solution in your favorite C# IDE. I recommend JetBrains Rider, but Visual Studio should work as well.

There is currently no way to run the project outside of the API, as unit tests are a work in progress.

### Creating a dataset

### Running
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
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

The Galaxypedia itself is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License. Please ensure you are compliant with the license when using the dataset.

In general, I would appreciate that you credit the Galaxypedia if you create derivative works, or if you decide to use the dataset.
