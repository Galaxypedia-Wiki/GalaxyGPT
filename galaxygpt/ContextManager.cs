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
        QdrantClient qdrantClient)
    {
        _embeddingClient = embeddingClient;
        _embeddingsTokenizer = embeddingsTokenizer;
        _qdrantClient = qdrantClient;
    }

    /// <summary>
    /// Get the context for a question
    /// </summary>
    /// <param name="question">The question to get the context for</param>
    /// <param name="maxResults">How many results to fetch. Defaults to top 5</param>
    /// <returns>A tuple containing the context and the amount of embedding tokens the question used up</returns>
    /// <exception cref="InvalidOperationException"></exception>
    /// <exception cref="ArgumentException"></exception>
    public virtual async Task<(string context, int questiontokens)> FetchContext(string question, ulong maxResults = 5)
    {
        if (_qdrantClient == null)
            throw new InvalidOperationException("The Qdrant client is not available.");

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
            context
                .AppendLine($"Page: {searchResult.Payload["title"].StringValue}")
                .AppendLine($"Content: {searchResult.Payload["content"]}")
                .AppendLine()
                .AppendLine()
                .AppendLine("###")
                .AppendLine()
                .AppendLine();
        }

        return (context.ToString(), _embeddingsTokenizer.CountTokens(question));
    }
}