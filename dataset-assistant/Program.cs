using System.ClientModel;
using System.CommandLine;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using galaxygpt;
using galaxygpt.Database;
using Microsoft.EntityFrameworkCore;
using OpenAI.Embeddings;

namespace dataset_assistant;

partial class Program
{
    // Gallery tag regex
    [GeneratedRegex(@"(\|image.?=.?)?<gallery.*?>.*?<\/gallery>\\?\n?", RegexOptions.Singleline)]
    private static partial Regex GalleryTagRegex();

    // File link regex
    [GeneratedRegex(@"\[\[File:.*?\]\]\\?", RegexOptions.Singleline)]
    private static partial Regex FileLinkRegex();

    // Magic word regex
    [GeneratedRegex(@"__.*?__", RegexOptions.Singleline)]
    private static partial Regex MagicWordRegex();

    // HTML comments regex
    [GeneratedRegex(@"<!--.*?-->\\?\n?", RegexOptions.Singleline)]
    private static partial Regex HtmlCommentRegex();

    // Span & br regex
    [GeneratedRegex(@"<span.*?>|<\/span>\\?\n?|<br.*?>\\?\n?", RegexOptions.Singleline)]
    private static partial Regex SpanBrRegex();

    // Div tags regex
    [GeneratedRegex(@"<div.*?>|<\/div>\\?\n?", RegexOptions.Singleline)]
    private static partial Regex DivTagRegex();

    private static async Task<int> Main(string[] args)
    {
        #region Options

        var datasetDirectory = new Option<string>(
            ["--directory", "-d"],
            "The directory that stores the datasets"
        )
        {
            IsRequired = true
        };

        var cleanDirOption = new Option<bool>(
            ["--cleanDir", "-c"],
            "Clean the output directory before writing the dataset"
        );

        var noEmbeddingsOption = new Option<bool>(
            ["--noEmbeddings", "-n"],
            "Do not include embeddings in the dataset"
        );

        var dumpDatabaseOption = new Option<bool>(
            ["--dumpDatabase", "-dd"],
            "Dump the database to the output directory"
        );

        var maxLengthOption = new Option<int>(
            ["--maxLength", "-m"],
            "The maximum length of the content"
        )
        {
            IsRequired = true
        };

        var compressOldDatasetsOption = new Option<bool>(
            ["--compressOldDatasets", "-C"],
            "Compress old datasets in the output directory"
        );

        var datasetNameOption = new Option<string>(
            ["--datasetName", "-N"],
            "The name of the dataset"
        )
        {
            IsRequired = true
        };

        var dumpPathOption = new Option<string>(
            ["--dbDumpPath", "-D"],
            "The path to the database dump"
        )
        {
            IsRequired = true
        };

        #endregion

        var rootCommand = new RootCommand("GalaxyGPT Dataset Management Assistant")
        {
            datasetDirectory,
            cleanDirOption,
            noEmbeddingsOption,
            dumpDatabaseOption,
            maxLengthOption,
            compressOldDatasetsOption,
            datasetNameOption,
            dumpPathOption
        };

        rootCommand.SetHandler(async handler =>
        {
            string? datasetDirectoryValue = handler.ParseResult.GetValueForOption(datasetDirectory);
            bool? cleanDirOptionValue = handler.ParseResult.GetValueForOption(cleanDirOption);
            bool? noEmbeddingsOptionValue = handler.ParseResult.GetValueForOption(noEmbeddingsOption);
            bool? dumpDatabaseOptionValue = handler.ParseResult.GetValueForOption(dumpDatabaseOption);
            int? maxLengthOptionValue = handler.ParseResult.GetValueForOption(maxLengthOption);
            bool? compressOldDatasetsOptionValue = handler.ParseResult.GetValueForOption(compressOldDatasetsOption);
            string? datasetNameOptionValue = handler.ParseResult.GetValueForOption(datasetNameOption);
            string dumpPathOptionValue = handler.ParseResult.GetValueForOption(dumpPathOption)!;

            await GalaxyGpt.Db.Database.EnsureDeletedAsync();
            await GalaxyGpt.Db.Database.MigrateAsync();

            string csvData = await File.ReadAllTextAsync(dumpPathOptionValue);
            csvData = csvData.Replace("\\\n", "\n");

            // Read the database dump, which is a csv
            using var reader = new StringReader(csvData);
            using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                BadDataFound = null,
                MissingFieldFound = null,
                TrimOptions = TrimOptions.Trim,
                Encoding = Encoding.UTF8,
                Escape = '\\',
                Quote = '"',
                NewLine = Environment.NewLine,
                Mode = CsvMode.RFC4180,
                AllowComments = false
            });

            await csv.ReadAsync();
            csv.ReadHeader();

            // Could possibly run some of this in parallel
            while (await csv.ReadAsync())
            {
                string? title = csv.GetField<string>("page_title");
                string? content = csv.GetField<string>("content");

                if (string.IsNullOrEmpty(title) || string.IsNullOrEmpty(content))
                    continue;

                // Page title sanitization
                title = title.Replace("_", " ").Trim();

                // Content sanitization
                content = content.Replace("\n", " ");
                content = GalleryTagRegex().Replace(content, "");
                content = FileLinkRegex().Replace(content, "");
                content = MagicWordRegex().Replace(content, "");
                content = HtmlCommentRegex().Replace(content, "");
                content = SpanBrRegex().Replace(content, "");
                content = DivTagRegex().Replace(content, "");
                content = content.Trim();

                if (string.IsNullOrEmpty(title) || string.IsNullOrEmpty(content))
                    continue;

                var page = new Page
                {
                    Title = title,
                    Content = content
                };

                GalaxyGpt.Db.Add(page);
            }

            await GalaxyGpt.Db.SaveChangesAsync();

            // Finished adding all the pages to the database.

            const int maxtokens = 8192;

            // Chunk the pages into smaller pages
            foreach (Page page in GalaxyGpt.Db.Pages)
            {
                List<Chunk> chunks = [];
                string content = page.Content;

                if (page.Tokens <= maxtokens) continue;
                while (true) // Loop until the content is empty
                {
                    int splitIndex = GalaxyGpt.EmbeddingsTokenizer.GetIndexByTokenCount(content, maxtokens, out string? _, out int tokencount);
                    string chunk = content[..splitIndex];
                    Console.WriteLine("Splitting page " + page.Title + " at index " + splitIndex + " with token count " + tokencount);
                    chunks.Add(new Chunk { Content = chunk });

                    // The last chunk will be the remainder of the content. So we break the loop here
                    if (splitIndex == content.Length)
                        break;

                    content = content[splitIndex..];
                }

                page.Chunks = chunks;
            }

            await GalaxyGpt.Db.SaveChangesAsync();

            // Create embeddings for each page, or chunks if they exist. Can also be done in parallel
            EmbeddingClient? embeddingsClient = GalaxyGpt.OpenAiClient.GetEmbeddingClient("text-embedding-3-small");

            await Parallel.ForEachAsync(GalaxyGpt.Db.Pages, async (page, cancellationToken) =>
            {
                // Handle the case where the page has no chunks
                if (page.Chunks == null || page.Chunks.Count == 0)
                {
                    Console.WriteLine("generating embeddings for " + page.Title);
                    ClientResult<Embedding>? embedding = await embeddingsClient.GenerateEmbeddingAsync(page.Content, cancellationToken: cancellationToken);
                    page.Embeddings = embedding.Value.Vector.ToArray().ToList();
                    return;
                }

                int chunkNumber = 0;
                // Handle the case where the page has chunks
                foreach (Chunk chunk in page.Chunks)
                {
                    Console.WriteLine($"generating embeddings for {page.Title} chunk {chunkNumber} with token count {GalaxyGpt.GptTokenizer.CountTokens(chunk.Content)}");
                    ClientResult<Embedding>? embedding = await embeddingsClient.GenerateEmbeddingAsync(chunk.Content, cancellationToken: cancellationToken);
                    chunk.Embeddings = embedding.Value.Vector.ToArray().ToList();
                    chunkNumber++;
                }
            });

            await GalaxyGpt.Db.SaveChangesAsync();
        });

        return await rootCommand.InvokeAsync(args);
    }
}