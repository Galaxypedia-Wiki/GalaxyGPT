// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.CommandLine;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using galaxygpt.Database;
using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML.Tokenizers;
using OpenAI;
using OpenAI.Embeddings;

namespace dataset_assistant;

partial class Program
{
    #region Regex

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

    #endregion

    private static async Task<int> Main(string[] args)
    {
        #region Options

        var datasetDirectory = new Option<string>(
            ["--directory", "-d"],
            getDefaultValue: () => ".",
            "The directory that stores the dataset folders (i.e. working directory)"
        )
        {
            IsRequired = true
        };

        var cleanDirOption = new Option<bool>(
            ["--cleanDir", "-c"],
            "If a dataset with the same name already exists, delete it before proceeding"
        );

        var noEmbeddingsOption = new Option<bool>(
            ["--noEmbeddings", "-n"],
            "Do not generate embeddings for the dataset. Useful for debugging without incurring API costs"
        );

        var compressOldDatasetsOption = new Option<bool>(
            ["--compressOldDatasets", "-C"],
            "Compress old datasets in the output directory"
        );

        var datasetNameOption = new Option<string>(
            ["--datasetName", "-N"],
            "The name of the dataset. Defaults to dataset-v<n> where <n> is an increment of the latest dataset in the directory"
        );

        var dumpPathOption = new Option<string>(
            ["--dbDumpPath", "-D"],
            "The path to the database dump file. This file should be a csv file"
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
            bool? compressOldDatasetsOptionValue = handler.ParseResult.GetValueForOption(compressOldDatasetsOption);
            string? datasetNameOptionValue = handler.ParseResult.GetValueForOption(datasetNameOption);
            string dumpPathOptionValue = handler.ParseResult.GetValueForOption(dumpPathOption)!;
            string embeddingsModelOptionValue = handler.ParseResult.GetValueForOption(embeddingsModelOption)!;
            string? openAiApiKeyOptionValue = handler.ParseResult.GetValueForOption(openAiApiKeyOption);

            #endregion

            #region Directory Validation

            if (!Directory.Exists(datasetDirectoryValue))
                throw new InvalidOperationException("The dataset directory does not exist");

            string[] existingDatasets = Directory.GetDirectories(datasetDirectoryValue, "dataset-v*");

            datasetNameOptionValue ??= existingDatasets.Length == 0
                ? "dataset-v1"
                : $"dataset-v{existingDatasets.Length + 2}";

            if (cleanDirOptionValue == true && Directory.Exists(Path.Combine(datasetDirectoryValue, datasetNameOptionValue)))
                Directory.Delete(Path.Combine(datasetDirectoryValue, datasetNameOptionValue), true);

            if (compressOldDatasetsOptionValue == true)
            {
                foreach (string dataset in existingDatasets)
                {
                    string zipPath = Path.Combine(datasetDirectoryValue, $"{Path.GetFileName(dataset)}.zip");
                    ZipFile.CreateFromDirectory(dataset, zipPath);
                    Directory.Delete(dataset, true);
                }
            }

            Directory.CreateDirectory(Path.Combine(datasetDirectoryValue, datasetNameOptionValue));

            #endregion

            #region Dependencies

            Console.WriteLine("Setting up dependencies");
            await using var db = new VectorDb(Path.Combine(datasetDirectoryValue, datasetNameOptionValue));
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
                    int splitIndex = embeddingsTokenizer.GetIndexByTokenCount(content, maxtokens, out string? _, out int _);
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

            if (noEmbeddingsOptionValue == true)
                return;

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

        var legacyDumpCommand = new Option<bool>(
            "--legacy",
            "Use the legacy database dump method. Which is using the mysql command line tool instead of the .NET connector"
        );

        var dumpCommand = new Command("dump", "Dump the database to the output directory")
        {
            legacyDumpCommand
        };
        rootCommand.AddCommand(dumpCommand);

        dumpCommand.SetHandler(async handler =>
        {
            Console.Write("Enter the database user password: ");
            string? password = Console.ReadLine();

            if (string.IsNullOrEmpty(password))
                throw new InvalidOperationException("Password cannot be empty");

            if (!handler.ParseResult.GetValueForOption(legacyDumpCommand))
            {
                SqlConnectionStringBuilder builder = new()
                {
                    DataSource = "localhost",
                    InitialCatalog = "galaxypedia",
                    UserID = "root",
                    Password = password
                };

                await using SqlConnection connection = new(builder.ConnectionString);
                await connection.OpenAsync();

                // Select the page_namespace, page_title, and old_text (content) columns from the page table
                var command = new SqlCommand(
                    """SELECT page_namespace, page_title "page_name", old_text "content" FROM page INNER JOIN slots on page_latest = slot_revision_id INNER JOIN slot_roles on slot_role_id = role_id AND role_name = 'main' INNER JOIN content on slot_content_id = content_id INNER JOIN text on substring( content_address, 4 ) = old_id AND left( content_address, 3 ) = "tt:" WHERE (page.page_namespace = 0 OR page.page_namespace = 4) AND page.page_is_redirect = 0""",
                    connection);
                await using SqlDataReader reader = await command.ExecuteReaderAsync();

                await using var csvWriter = new CsvWriter(new StreamWriter("dump.csv", false, Encoding.UTF8),
                    new CsvConfiguration(CultureInfo.InvariantCulture)
                    {
                        HasHeaderRecord = true,
                    });

                // Add the header row
                csvWriter.WriteField("page_namespace");
                csvWriter.WriteField("page_title");
                csvWriter.WriteField("content");

                await csvWriter.WriteRecordsAsync(reader);

                Console.WriteLine("Database dumped to dump.csv");

                reader.Close();
                await connection.CloseAsync();
            }
            else
            {
                const string sqlDumpQuery =
                    """USE galaxypedia; SELECT page_namespace, page_title "page_name", old_text "content" FROM page INNER JOIN slots on page_latest = slot_revision_id INNER JOIN slot_roles on slot_role_id = role_id AND role_name = 'main' INNER JOIN content on slot_content_id = content_id INNER JOIN text on substring( content_address, 4 ) = old_id AND left( content_address, 3 ) = "tt:" WHERE (page.page_namespace = 0 OR page.page_namespace = 4) AND page.page_is_redirect = 0 into outfile '/tmp/galaxypedia.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';""";
                using Process? process = Process.Start(new ProcessStartInfo
                {
                    FileName = "mysql",
                    Arguments = $"-u root -p{password} -e {sqlDumpQuery}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = true
                });

                if (process == null)
                    throw new InvalidOperationException("Failed to start the mysql process");

                await process.WaitForExitAsync();

                Console.WriteLine("Database dumped to /tmp/galaxypedia.csv");

                File.Move("/tmp/galaxypedia.csv", "dump.csv", true);

                string username = Environment.UserName;
                using Process? chownProcess = Process.Start(new ProcessStartInfo
                {
                    FileName = "chown",
                    Arguments = $"{username}:{username} dump.csv",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = true
                });

                if (chownProcess == null)
                    throw new InvalidOperationException("Failed to start the chown process");

                await chownProcess.WaitForExitAsync();
                Console.WriteLine("Done");
            }
        });

        return await rootCommand.InvokeAsync(args);
    }
}
