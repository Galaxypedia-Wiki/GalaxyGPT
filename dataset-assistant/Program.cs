﻿// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.CommandLine;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.Data.SqlClient;
using Microsoft.ML.Tokenizers;
using OpenAI;
using OpenAI.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using ShellProgressBar;

namespace dataset_assistant;

internal partial class Program
{
    private static async Task<int> Main(string[] args)
    {
        #region Options

        var dumpPathOption = new Option<string>(
            ["--dbDumpPath", "-D"],
            "The path to the database dump file. This file should be a csv file"
        )
        {
            IsRequired = true
        };

        var embeddingsModelOption = new Option<string>(
            ["--embeddingsModel", "-e"],
            () => "text-embedding-3-small",
            "The embeddings model to use"
        );

        var openAiApiKeyOption = new Option<string>(
            ["--openAiApiKey", "-k"],
            "The OpenAI API key"
        );

        var qdrantUrlOption = new Option<string>(
            ["--qdrantUrl", "-q"],
            () => "localhost:6334",
            "The URL of the Qdrant server"
        );

        #endregion

        var rootCommand = new RootCommand("GalaxyGPT Dataset Management Assistant")
        {
            qdrantUrlOption,
            dumpPathOption,
            embeddingsModelOption,
            openAiApiKeyOption
        };

        rootCommand.SetHandler(async handler =>
        {
            #region Option Values

            string qdrantUrlOptionValue = handler.ParseResult.GetValueForOption(qdrantUrlOption)!;
            string dumpPathOptionValue = handler.ParseResult.GetValueForOption(dumpPathOption)!;
            string embeddingsModelOptionValue = handler.ParseResult.GetValueForOption(embeddingsModelOption)!;
            string? openAiApiKeyOptionValue = handler.ParseResult.GetValueForOption(openAiApiKeyOption);

            #endregion

            using var globalProgressBar = new ProgressBar(7, "Validating Directories");

            #region Dependencies

            globalProgressBar.Tick("Resolving Dependencies");
            var embeddingsTokenizer = TiktokenTokenizer.CreateForModel(embeddingsModelOptionValue);
            OpenAIClient openAiClient = new(openAiApiKeyOptionValue ??
                                            Environment.GetEnvironmentVariable("OPENAI_API_KEY") ??
                                            throw new InvalidOperationException());

            string[] qdrantUrlAndPort = qdrantUrlOptionValue.Split(':');
            var qdrantClient = new QdrantClient(qdrantUrlAndPort[0], qdrantUrlAndPort.Length == 2
                ? int.Parse(qdrantUrlAndPort[1])
                : 6334);

            await qdrantClient.RecreateCollectionAsync("galaxypedia", new VectorParams
            {
                Distance = Distance.Cosine,
                Size = 1536
            });

            #endregion

            #region CSV Reading

            globalProgressBar.Tick("Reading Database Dump");

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

            List<(string, string)> pages = [];

            #region CSV Sanitization & Database Insertion

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

                pages.Add((title, content));
            }

            #endregion

            #region Chunking

            const int maxtokens = 8192;

            globalProgressBar.Tick("Chunking Pages");
            using ChildProgressBar? chunkingProgressBar = globalProgressBar.Spawn(pages.Count, "Chunking Pages", new ProgressBarOptions());

            // Holds the page title, chunk content, and chunk token count
            var chunksList = new List<(string, string, int)>();

            foreach ((string, string) page in pages)
            {
                List<string> chunks = [];
                string content = page.Item2;

                while (content.Length > 0)
                {
                    int splitIndex =
                        embeddingsTokenizer.GetIndexByTokenCount(content, maxtokens, out string? _, out int _);
                    chunks.Add(content[..splitIndex]);

                    if (splitIndex == content.Length)
                        break;

                    content = content[splitIndex..];
                }

                chunksList.AddRange(chunks.Select(chunk => (page.Item1, chunk, embeddingsTokenizer.CountTokens(chunk))));
                chunkingProgressBar.Tick();
            }

            #endregion

            #region Embedding

            globalProgressBar.Tick("Generating Embeddings");
            using ChildProgressBar? embeddingsProgressBar =
                globalProgressBar.Spawn(chunksList.Count, "Generating Embeddings");

            EmbeddingClient? embeddingsClient = openAiClient.GetEmbeddingClient(embeddingsModelOptionValue);

            var embeddedChunks = new List<(string, string, int, float[])>();

            foreach ((string, string, int) chunk in chunksList)
            {
                embeddedChunks.Add((chunk.Item1, chunk.Item2, chunk.Item3,
                    (await embeddingsClient.GenerateEmbeddingAsync(chunk.Item2)).Value.Vector.ToArray()));
                embeddingsProgressBar.Tick();
            }

            #endregion

            globalProgressBar.Tick("Upserting Points");
            List<PointStruct> points = [];
            points.AddRange(embeddedChunks.Select(embeddedChunk => new PointStruct
            {
                Id = Guid.NewGuid(),
                Vectors = embeddedChunk.Item4,
                Payload =
                {
                    ["title"] = embeddedChunk.Item1,
                    ["content"] = embeddedChunk.Item2,
                    ["tokens"] = embeddedChunk.Item3
                }
            }));

            // Add the points to the Qdrant collection
            await qdrantClient.UpsertAsync("galaxypedia", points);

            globalProgressBar.Tick("Taking Snapshot");
            // Create a snapshot of the collection
            await qdrantClient.CreateSnapshotAsync("galaxypedia");

            globalProgressBar.Tick("Done");
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
                        HasHeaderRecord = true
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
}