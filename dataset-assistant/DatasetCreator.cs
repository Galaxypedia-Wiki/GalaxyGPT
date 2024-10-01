// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.ML.Tokenizers;
using OpenAI;
using OpenAI.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using Spectre.Console;

namespace dataset_assistant;

/// <summary>
/// Class that provides high level methods (chunking, embedding, etc.) for creating datasets
/// </summary>
public static partial class DatasetCreator
{
    public static async Task UpsertPointsIntoQdrant(List<(string, string, int, float[])> embeddedChunks, QdrantClient qdrantClient)
    {
        await qdrantClient.UpsertAsync("galaxypedia", embeddedChunks.Select(embeddedChunk => new PointStruct
        {
            Id = Guid.NewGuid(),
            Vectors = embeddedChunk.Item4,
            Payload =
            {
                ["title"] = embeddedChunk.Item1,
                ["content"] = embeddedChunk.Item2,
                ["tokens"] = embeddedChunk.Item3
            }
        }).ToList());

        await qdrantClient.CreateSnapshotAsync("galaxypedia");
    }

    public static async Task<List<(string title, string content, int tokencount, float[] embeddings)>> GenerateEmbeddedChunks(List<(string title, string content, int tokencount)> chunksList, ProgressTask embeddingTask, OpenAIClient openAiClient,
        string embeddingsModelOptionValue)
    {
        EmbeddingClient? embeddingsClient = openAiClient.GetEmbeddingClient(embeddingsModelOptionValue);

        var embeddedChunks = new List<(string title, string content, int tokencount, float[] embeddings)>();

        // Warning: This could reach rate limits. The request per minute limit is 3000-5000 for tier 2 users which is the average tier, so we should be fine.
        // Might be wise to allow limiting it via an environment variable for new api users.
        await Parallel.ForEachAsync(chunksList, async (chunk, token) =>
        {
            embeddedChunks.Add(
                (
                    chunk.title,
                    chunk.content,
                    chunk.tokencount,
                    (await embeddingsClient.GenerateEmbeddingAsync(chunk.Item2,
                        cancellationToken: token)).Value.ToFloats().ToArray())
            );
            embeddingTask.Increment(1);
        });

        return embeddedChunks;
    }

    public static List<(string title, string content, int tokencount)> ChunkPages(List<(string, string)> pages, ProgressTask chunkingTask, TiktokenTokenizer embeddingsTokenizer)
    {
        chunkingTask.MaxValue(pages.Count);
        // Holds the page title, chunk content, and chunk token count
        var chunksList = new List<(string title, string content, int tokencount)>();

        foreach ((string title, string content) page in pages)
        {
            List<string> chunks = [];
            string content = page.content;

            while (content.Length > 0)
            {
                int splitIndex =
                    embeddingsTokenizer.GetIndexByTokenCount(content, Program.Maxtokens, out string? _, out int _);
                chunks.Add(content[..splitIndex]);

                if (splitIndex == content.Length)
                    break;

                content = content[splitIndex..];
            }

            // It's a little misleading. The chunks belong to the same page. But at the same time, we need the pages to be unique (i.e. in batching)
            // For now, I've decided to append the index to the page title to make it unique if there are multiple chunks
            chunksList.AddRange(chunks.Select((chunk, index) => (page.Item1 + (chunks.Count > 1 ? $" ({index+1})" : ""), chunk, embeddingsTokenizer.CountTokens(chunk))));
            chunkingTask.Increment(1);
        }

        return chunksList;
    }

    public static async Task<List<(string title, string content)>> GetPagesFromCsv(string csvFile)
    {
        #region CSV Reading

        string csvData = await File.ReadAllTextAsync(csvFile);
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

        // I wonder why we use tuples instead of a dictionary here, feels like it'd be more readable
        List<(string title, string content)> pages = [];

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

        return pages;
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