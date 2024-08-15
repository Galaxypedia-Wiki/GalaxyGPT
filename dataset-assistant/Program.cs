// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.CommandLine;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using galaxygpt.Database;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML.Tokenizers;
using OpenAI;
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

    [GeneratedRegex(@"'{3,}(.*?)'{3,}")]
    private static partial Regex BoldItalicsRegex();

    [GeneratedRegex(@"\[\[([^\[\]\|]+?)\|([^\[\]]+?)\]\]")]
    private static partial Regex LinkSlicerRegex();

    [GeneratedRegex(@"\[\[(.*?)\]\]")]
    private static partial Regex ShortenLinksRegex();

    [GeneratedRegex(@"\s+")]
    private static partial Regex ExtraWhitespaceRegex();

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

        var embeddingsModelOption = new Option<string>(
            ["--embeddingsModel", "-e"],
            getDefaultValue: () => "text-embedding-3-small",
            "The embeddings model to use"
        );

        var openAiApiKeyOption = new Option<string>(
            ["--openAiApiKey", "-k"],
            "The OpenAI API key"
        );

        #endregion

        var rootCommand = new RootCommand("GalaxyGPT Dataset Management Assistant")
        {
            datasetDirectory,
            cleanDirOption,
            noEmbeddingsOption,
            dumpDatabaseOption,
            compressOldDatasetsOption,
            datasetNameOption,
            dumpPathOption,
            embeddingsModelOption,
            openAiApiKeyOption
        };

        rootCommand.SetHandler(async handler =>
        {
            #region Option Values

            string? datasetDirectoryValue = handler.ParseResult.GetValueForOption(datasetDirectory);
            bool? cleanDirOptionValue = handler.ParseResult.GetValueForOption(cleanDirOption);
            bool? noEmbeddingsOptionValue = handler.ParseResult.GetValueForOption(noEmbeddingsOption);
            bool? dumpDatabaseOptionValue = handler.ParseResult.GetValueForOption(dumpDatabaseOption);
            bool? compressOldDatasetsOptionValue = handler.ParseResult.GetValueForOption(compressOldDatasetsOption);
            string? datasetNameOptionValue = handler.ParseResult.GetValueForOption(datasetNameOption);
            string dumpPathOptionValue = handler.ParseResult.GetValueForOption(dumpPathOption)!;
            string embeddingsModelOptionValue = handler.ParseResult.GetValueForOption(embeddingsModelOption)!;
            string? openAiApiKeyOptionValue = handler.ParseResult.GetValueForOption(openAiApiKeyOption);

            #endregion

            #region Dependencies

            Console.WriteLine("Setting up dependencies");
            await using var db = new VectorDb();
            var embeddingsTokenizer = TiktokenTokenizer.CreateForModel(embeddingsModelOptionValue);
            OpenAIClient openAiClient = new(openAiApiKeyOptionValue ?? Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new InvalidOperationException());

            await db.Database.EnsureDeletedAsync();
            await db.Database.MigrateAsync();

            #endregion

            #region CSV Reading

            string csvData = await File.ReadAllTextAsync(dumpPathOptionValue);
            csvData = csvData.Replace("\\\n", "\n");

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

            #endregion

            #region CSV Sanitization & Database Insertion

            Console.WriteLine("Sanitizing and inserting data into the database");
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
                content = BoldItalicsRegex().Replace(content, "$1");
                content = LinkSlicerRegex().Replace(content, "$2");
                content = ShortenLinksRegex().Replace(content, "$1");
                content = ExtraWhitespaceRegex().Replace(content, " ");
                content = content.Trim();

                if (string.IsNullOrEmpty(title) || string.IsNullOrEmpty(content))
                    continue;

                var page = new Page
                {
                    Title = title,
                    Content = content
                };

                db.Add(page);
            }

            await db.SaveChangesAsync();

            #endregion

            #region Chunking

            const int maxtokens = 8192;

            Console.WriteLine("Chunking pages");
            foreach (Page page in db.Pages)
            {
                if (page.Tokens <= maxtokens) continue;

                List<Chunk> chunks = [];
                string content = page.Content;

                while (content.Length > 0)
                {
                    int splitIndex = embeddingsTokenizer.GetIndexByTokenCount(content, maxtokens, out string? _, out int tokencount);
                    chunks.Add(new Chunk { Content = content[..splitIndex] });

                    if (splitIndex == content.Length)
                        break;

                    content = content[splitIndex..];
                }

                page.Chunks = chunks;
            }

            await db.SaveChangesAsync();

            #endregion

            #region Embedding

            EmbeddingClient? embeddingsClient = openAiClient.GetEmbeddingClient(embeddingsModelOptionValue);

            Console.WriteLine("Generating embeddings");
            foreach (Page page in db.Pages.Include(page => page.Chunks))
            {
                Console.WriteLine("Embedding page " + page.Title);
                // Handle the case where the page has no chunks
                if (page.Chunks == null || page.Chunks.Count == 0)
                {
                    ClientResult<Embedding>? embedding = await embeddingsClient.GenerateEmbeddingAsync(page.Content);
                    page.Embeddings = embedding.Value.Vector.ToArray().ToList();
                }
                else
                {
                    // Handle the case where the page has chunks
                    foreach (Chunk chunk in page.Chunks)
                    {
                        ClientResult<Embedding>? embedding = await embeddingsClient.GenerateEmbeddingAsync(chunk.Content);
                        chunk.Embeddings = embedding.Value.Vector.ToArray().ToList();
                    }
                }
            }

            await db.SaveChangesAsync();

            #endregion
        });

        return await rootCommand.InvokeAsync(args);
    }
}
