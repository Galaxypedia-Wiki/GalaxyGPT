// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Tokenizers;
using OpenAI.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace galaxygpt;

/// <summary>
///     Handles context management
/// </summary>
public class ContextManager
{
    private readonly QdrantClient? _qdrantClient;

    private readonly EmbeddingClient _embeddingClient;
    private readonly TiktokenTokenizer _embeddingsTokenizer;

    public ContextManager(EmbeddingClient embeddingClient,
        [FromKeyedServices("embeddingsTokenizer")]
        TiktokenTokenizer embeddingsTokenizer,
        string? qdrantUrl)
    {
        _embeddingClient = embeddingClient;
        _embeddingsTokenizer = embeddingsTokenizer;
        string[] qdrantUrlAndPort = (qdrantUrl ?? "qdrant").Split(":");

        // TODO: Move to DI (its impossible to test this class because we can't mock QdrantClient)
        // Okay turns out this is probably going to be a bit more difficult than I thought. It doesn't seem like
        // QDrantClient can be mocked (it doesnt expose any virtual members). We'd probably have to create a method for
        // running the query and then mock that method instead of QdrantClient's methods.
        _qdrantClient = new QdrantClient(qdrantUrlAndPort[0],
            qdrantUrlAndPort.Length > 1 ? int.Parse(qdrantUrlAndPort[1]) : 6334);
    }

    public async Task<(string context, int contextTokenCount, int questionTokenCount)> FetchContext(string question, ulong maxResults = 5)
    {
        if (_qdrantClient == null)
            throw new InvalidOperationException("The Qdrant client is not available.");
        question = question.Trim();

        if (string.IsNullOrWhiteSpace(question))
            throw new ArgumentException("The question cannot be empty.");

        ClientResult<Embedding>? questionEmbeddings = await _embeddingClient.GenerateEmbeddingAsync(question);

        IReadOnlyList<ScoredPoint> searchResults = await _qdrantClient.QueryAsync(
            "galaxypedia",
            questionEmbeddings.Value.ToFloats().ToArray(),
            limit: maxResults,
            payloadSelector: true
        );

        StringBuilder context = new();

        foreach (ScoredPoint searchResult in searchResults)
        {
            context.Append($"Page: {searchResult.Payload["title"].StringValue}\nContent: {searchResult.Payload["content"].StringValue}\n\n###\n\n");
        }

        string contextString = context.ToString();
        return (contextString, _embeddingsTokenizer.CountTokens(contextString), _embeddingsTokenizer.CountTokens(question));
    }
}